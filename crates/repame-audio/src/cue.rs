#[derive(Clone, Debug, PartialEq)]
pub struct Cue {
    pub name: &'static str,
    pub volume: f32,
    pub variance: f32,
}

impl Cue {
    pub fn pitch(&self, rng01: f32) -> f32 {
        1.0 + (rng01 * 2.0 - 1.0) * self.variance
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn variance_centers_on_one() {
        let c = Cue {
            name: "s",
            volume: 0.8,
            variance: 0.1,
        };
        assert!((c.pitch(0.5) - 1.0).abs() < 1e-6);
    }
}
