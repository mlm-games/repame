//! Typed errors returned by the solvers and modifiers.

/// Why an IK or modifier operation could not run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IkError {
    /// The joint chain has no joints.
    EmptyChain,
    /// A joint index is not a node in the skeleton.
    UnknownJoint(usize),
    /// `child` is not a descendant of `parent`, so no chain connects them.
    BrokenChain { parent: usize, child: usize },
}

impl std::fmt::Display for IkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IkError::EmptyChain => write!(f, "joint chain is empty"),
            IkError::UnknownJoint(node) => write!(f, "joint {node} is not in the skeleton"),
            IkError::BrokenChain { parent, child } => {
                write!(f, "joint {child} is not a descendant of joint {parent}")
            }
        }
    }
}

impl std::error::Error for IkError {}
