//! Packet and frame layout shared by both transports.
//!
//! # UDP datagram
//!
//! ```text
//! datagram := version:u8, count:u8, record:count
//! record   := tag:u8, body
//!   tag 0 ACK        : channel:u8, ack_seq:u32, ack_bits:u32
//!   tag 1 RELIABLE   : channel:u8, seq:u32, payload:bytes
//!   tag 2 UNRELIABLE : channel:u8, payload:bytes
//! ```
//!
//! `ack_bits` bit `i` means the receiver holds `ack_seq - 1 - i`. The receiver's
//! reliability for the channel decides whether RELIABLE records are held in
//! order (see [`crate::transport::udp::ReliableEndpoint`]). A record whose
//! reliability class contradicts its channel is a
//! [`crate::NetError::Protocol`], never a silent drop.
//!
//! # TCP frame
//!
//! ```text
//! frame := length:u32, body:bytes[length]
//! body  := version:u8, count:u8, message:count
//! message := channel:u8, payload:bytes
//! ```
//!
//! TCP already guarantees reliable, ordered delivery, so frames carry no
//! sequence numbers and every channel behaves as reliable-ordered regardless
//! of the reliability recorded in the [`ChannelTable`](crate::ChannelTable);
//! that degradation is deliberate and documented here.

use crate::WIRE_VERSION;
use crate::channel::ChannelId;
use crate::codec::{Decode, Encode, Reader, Writer};
use crate::error::CodecError;

/// Largest TCP frame body accepted, in bytes (16 MiB).
pub const MAX_FRAME_BYTES: usize = 1 << 24;

/// Largest UDP datagram this crate emits, in bytes.
pub const MAX_DATAGRAM_BYTES: usize = 1200;

/// Size of the acknowledgement bitfield, in bits.
pub const ACK_WINDOW: u32 = 32;

/// Overhead of the largest record header: tag, channel, seq, length.
pub const MAX_RECORD_OVERHEAD: usize = 1 + 1 + 4 + 4;

/// Tag byte of a UDP record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecordTag {
    /// Cumulative acknowledgement of a channel's reliable sequence space.
    Ack,
    /// A reliable-recorded message with a sequence number.
    Reliable,
    /// A best-effort message with no sequence number.
    Unreliable,
}

impl RecordTag {
    /// Wire tag byte.
    pub fn byte(self) -> u8 {
        match self {
            Self::Ack => 0,
            Self::Reliable => 1,
            Self::Unreliable => 2,
        }
    }

    /// Parse a tag byte.
    pub fn from_byte(byte: u8) -> Result<Self, CodecError> {
        match byte {
            0 => Ok(Self::Ack),
            1 => Ok(Self::Reliable),
            2 => Ok(Self::Unreliable),
            other => Err(CodecError::UnknownTag(other)),
        }
    }
}

/// One record inside a UDP datagram.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Record {
    /// Acks every reliable sequence up to and including `ack_seq`, plus the
    /// window bits above it.
    Ack {
        /// Channel being acknowledged.
        channel: ChannelId,
        /// Highest sequence number held by the receiver.
        ack_seq: u32,
        /// Bits `0..32`, bit `i` = `ack_seq - 1 - i` also held.
        ack_bits: u32,
    },
    /// A message the sender retransmits until acknowledged.
    Reliable {
        /// Channel the message was sent on.
        channel: ChannelId,
        /// Per-channel, per-direction sequence number.
        seq: u32,
        /// Message bytes.
        payload: Vec<u8>,
    },
    /// A best-effort message: no sequence number, no retransmission.
    Unreliable {
        /// Channel the message was sent on.
        channel: ChannelId,
        /// Message bytes.
        payload: Vec<u8>,
    },
}

impl Record {
    /// Bytes this record occupies, including its tag.
    pub fn encoded_len(&self) -> usize {
        match self {
            Self::Ack { .. } => 1 + 1 + 4 + 4,
            Self::Reliable { payload, .. } => 1 + 1 + 4 + 4 + payload.len(),
            Self::Unreliable { payload, .. } => 1 + 1 + 4 + payload.len(),
        }
    }

    /// Channel the record belongs to.
    pub fn channel(&self) -> ChannelId {
        match self {
            Self::Ack { channel, .. }
            | Self::Reliable { channel, .. }
            | Self::Unreliable { channel, .. } => *channel,
        }
    }
}

impl Encode for Record {
    fn encode(&self, out: &mut Writer) -> Result<(), CodecError> {
        match self {
            Self::Ack {
                channel,
                ack_seq,
                ack_bits,
            } => {
                out.put_u8(RecordTag::Ack.byte());
                out.put_u8(*channel);
                out.put_u32(*ack_seq);
                out.put_u32(*ack_bits);
            }
            Self::Reliable {
                channel,
                seq,
                payload,
            } => {
                out.put_u8(RecordTag::Reliable.byte());
                out.put_u8(*channel);
                out.put_u32(*seq);
                out.put_bytes(payload)?;
            }
            Self::Unreliable { channel, payload } => {
                out.put_u8(RecordTag::Unreliable.byte());
                out.put_u8(*channel);
                out.put_bytes(payload)?;
            }
        }
        Ok(())
    }
}

impl Decode for Record {
    fn decode(input: &mut Reader<'_>) -> Result<Self, CodecError> {
        match RecordTag::from_byte(input.get_u8()?)? {
            RecordTag::Ack => Ok(Self::Ack {
                channel: input.get_u8()?,
                ack_seq: input.get_u32()?,
                ack_bits: input.get_u32()?,
            }),
            RecordTag::Reliable => Ok(Self::Reliable {
                channel: input.get_u8()?,
                seq: input.get_u32()?,
                payload: input.get_bytes()?.to_vec(),
            }),
            RecordTag::Unreliable => Ok(Self::Unreliable {
                channel: input.get_u8()?,
                payload: input.get_bytes()?.to_vec(),
            }),
        }
    }
}

/// Serialize a datagram of records with the leading version and count bytes.
pub fn encode_packet(records: &[Record]) -> Result<Vec<u8>, CodecError> {
    if records.len() > u8::MAX as usize {
        return Err(CodecError::Oversize {
            field: "record count",
            size: records.len(),
            max: u8::MAX as usize,
        });
    }
    let mut writer = Writer::with_capacity(2 + records.iter().map(Record::encoded_len).sum::<usize>());
    writer.put_u8(WIRE_VERSION);
    writer.put_u8(records.len() as u8);
    for record in records {
        record.encode(&mut writer)?;
    }
    Ok(writer.finish())
}

/// Parse a datagram; rejects a wrong version, unknown tag or trailing bytes.
pub fn decode_packet(bytes: &[u8]) -> Result<Vec<Record>, CodecError> {
    let mut reader = Reader::new(bytes);
    let version = reader.get_u8()?;
    if version != WIRE_VERSION {
        return Err(CodecError::BadVersion {
            found: version,
            expected: WIRE_VERSION,
        });
    }
    let count = reader.get_u8()?;
    let mut records = Vec::with_capacity(count as usize);
    for _ in 0..count {
        records.push(Record::decode(&mut reader)?);
    }
    reader.expect_end()?;
    Ok(records)
}

/// One channel message carried in a TCP frame body.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FramedMessage {
    /// Channel the message belongs to.
    pub channel: ChannelId,
    /// Message bytes.
    pub payload: Vec<u8>,
}

/// Encode a frame body (version, count, messages) without the length prefix.
pub fn encode_frame_body(messages: &[FramedMessage]) -> Result<Vec<u8>, CodecError> {
    if messages.len() > u8::MAX as usize {
        return Err(CodecError::Oversize {
            field: "message count",
            size: messages.len(),
            max: u8::MAX as usize,
        });
    }
    let mut writer = Writer::new();
    writer.put_u8(WIRE_VERSION);
    writer.put_u8(messages.len() as u8);
    for message in messages {
        writer.put_u8(message.channel);
        writer.put_bytes(&message.payload)?;
    }
    Ok(writer.finish())
}

/// Parse a frame body produced by [`encode_frame_body`].
pub fn decode_frame_body(body: &[u8]) -> Result<Vec<FramedMessage>, CodecError> {
    let mut reader = Reader::new(body);
    let version = reader.get_u8()?;
    if version != WIRE_VERSION {
        return Err(CodecError::BadVersion {
            found: version,
            expected: WIRE_VERSION,
        });
    }
    let count = reader.get_u8()?;
    let mut messages = Vec::with_capacity(count as usize);
    for _ in 0..count {
        let channel = reader.get_u8()?;
        let payload = reader.get_bytes()?.to_vec();
        messages.push(FramedMessage { channel, payload });
    }
    reader.expect_end()?;
    Ok(messages)
}

/// Encode a length-prefixed frame: `u32` body length then the body.
pub fn encode_frame(messages: &[FramedMessage]) -> Result<Vec<u8>, CodecError> {
    let body = encode_frame_body(messages)?;
    if body.len() > MAX_FRAME_BYTES {
        return Err(CodecError::Oversize {
            field: "frame",
            size: body.len(),
            max: MAX_FRAME_BYTES,
        });
    }
    let mut out = Vec::with_capacity(4 + body.len());
    out.extend_from_slice(&(body.len() as u32).to_le_bytes());
    out.extend_from_slice(&body);
    Ok(out)
}

/// Length of the body that follows a 4-byte frame prefix.
pub fn frame_length(prefix: [u8; 4]) -> Result<usize, CodecError> {
    let length = u32::from_le_bytes(prefix) as usize;
    if length > MAX_FRAME_BYTES {
        return Err(CodecError::Oversize {
            field: "frame",
            size: length,
            max: MAX_FRAME_BYTES,
        });
    }
    Ok(length)
}

/// Parse a complete length-prefixed frame, validating the prefix against the body.
pub fn decode_frame(frame: &[u8]) -> Result<Vec<FramedMessage>, CodecError> {
    if frame.len() < 4 {
        return Err(CodecError::UnexpectedEof {
            needed: 4,
            remaining: frame.len(),
        });
    }
    let length = frame_length([frame[0], frame[1], frame[2], frame[3]])?;
    let body_len = frame.len() - 4;
    if body_len < length {
        return Err(CodecError::UnexpectedEof {
            needed: length,
            remaining: body_len,
        });
    }
    if body_len > length {
        return Err(CodecError::TrailingBytes(body_len - length));
    }
    decode_frame_body(&frame[4..])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::CodecError;

    #[test]
    fn tcp_framing_round_trip() {
        let messages = vec![
            FramedMessage {
                channel: 0,
                payload: Vec::new(),
            },
            FramedMessage {
                channel: 2,
                payload: (0u8..=255).collect(),
            },
        ];
        let frame = encode_frame(&messages).unwrap();
        assert_eq!(
            &frame[..4],
            &(u32::try_from(frame.len() - 4).unwrap()).to_le_bytes()
        );
        assert_eq!(
            frame_length([frame[0], frame[1], frame[2], frame[3]]).unwrap(),
            frame.len() - 4
        );
        assert_eq!(decode_frame(&frame).unwrap(), messages);

        let mut bad_version = encode_frame_body(&messages).unwrap();
        bad_version[0] = WIRE_VERSION + 1;
        assert!(matches!(
            decode_frame_body(&bad_version),
            Err(CodecError::BadVersion { .. })
        ));
        assert!(decode_frame(&frame[..frame.len() - 1]).is_err());
    }
}
