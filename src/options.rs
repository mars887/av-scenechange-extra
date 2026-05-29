use std::{error::Error, fmt, str::FromStr};

use crate::{DetectionOptions, FastThresholdScale};

/// Returns the controlled list of option names accepted by
/// [`DetectionOptionOverride`]'s string parser.
#[inline]
#[must_use]
pub const fn detection_option_override_names() -> &'static [&'static str] {
    &[
        "flash-detection",
        "lookahead",
        "fast-threshold-scale",
        "forward-similarity",
        "forward-similarity-frames",
        "forward-similarity-min-offset",
        "forward-similarity-threshold",
        "forward-similarity-mask-percent",
        "forward-similarity-require-return-candidate",
        "forward-similarity-suppress-inside",
        "transient-similarity",
        "transient-similarity-frames",
        "transient-similarity-threshold",
        "transient-similarity-dark-threshold",
        "transient-similarity-mask-percent",
    ]
}

/// A controlled, typed override for scene detection internals.
///
/// This intentionally exposes only selected tuning switches instead of making
/// arbitrary internal fields configurable.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum DetectionOptionOverride {
    /// Enables or disables short flash suppression.
    FlashDetection(bool),
    /// Sets the base lookahead distance.
    Lookahead(usize),
    /// Selects how the fast detector threshold scales with bit depth.
    FastThresholdScale(FastThresholdScale),
    /// Enables or disables forward A-B-A similarity suppression.
    ForwardSimilarity(bool),
    /// Sets how many future frames forward similarity may inspect.
    ForwardSimilarityFrames(usize),
    /// Sets the minimum forward offset accepted as a return frame.
    ForwardSimilarityMinOffset(usize),
    /// Sets the accepted forward similarity luma delta in 8-bit units.
    ForwardSimilarityThreshold(f64),
    /// Sets the fraction of volatile blocks masked from forward similarity.
    ForwardSimilarityMaskPercent(f64),
    /// Requires a plausible future cut candidate before accepting a return.
    ForwardSimilarityRequireReturnCandidate(bool),
    /// Suppresses additional cuts until the matched forward return frame.
    ForwardSimilaritySuppressInside(bool),
    /// Enables or disables two-sided transient similarity suppression.
    TransientSimilarity(bool),
    /// Sets how many frames on each side transient similarity may inspect.
    TransientSimilarityFrames(usize),
    /// Sets the transient similarity threshold in 8-bit luma units.
    TransientSimilarityThreshold(f64),
    /// Sets the dark-scene transient similarity threshold in 8-bit luma units.
    TransientSimilarityDarkThreshold(f64),
    /// Sets the fraction of volatile blocks masked from transient similarity.
    TransientSimilarityMaskPercent(f64),
}

impl DetectionOptionOverride {
    #[inline]
    fn from_name_value(name: &str, value: &str) -> Result<Self, ParseDetectionOptionOverrideError> {
        match name {
            "flash-detection" => Ok(Self::FlashDetection(parse_bool(name, value)?)),
            "lookahead" => Ok(Self::Lookahead(parse_positive_usize(name, value)?)),
            "fast-threshold-scale" => Ok(Self::FastThresholdScale(match value {
                "legacy" => FastThresholdScale::Legacy,
                "sample-range" | "sample_range" => FastThresholdScale::SampleRange,
                _ => {
                    return Err(ParseDetectionOptionOverrideError::InvalidValue {
                        name: name.to_string(),
                        value: value.to_string(),
                        expected: "legacy|sample-range",
                    });
                }
            })),
            "forward-similarity" => Ok(Self::ForwardSimilarity(parse_bool(name, value)?)),
            "forward-similarity-frames" => {
                Ok(Self::ForwardSimilarityFrames(parse_usize(name, value)?))
            }
            "forward-similarity-min-offset" => {
                Ok(Self::ForwardSimilarityMinOffset(parse_usize(name, value)?))
            }
            "forward-similarity-threshold" => Ok(Self::ForwardSimilarityThreshold(
                parse_nonnegative_f64(name, value)?,
            )),
            "forward-similarity-mask-percent" => Ok(Self::ForwardSimilarityMaskPercent(
                parse_percent(name, value)?,
            )),
            "forward-similarity-require-return-candidate" => Ok(
                Self::ForwardSimilarityRequireReturnCandidate(parse_bool(name, value)?),
            ),
            "forward-similarity-suppress-inside" => Ok(Self::ForwardSimilaritySuppressInside(
                parse_bool(name, value)?,
            )),
            "transient-similarity" => Ok(Self::TransientSimilarity(parse_bool(name, value)?)),
            "transient-similarity-frames" => {
                Ok(Self::TransientSimilarityFrames(parse_usize(name, value)?))
            }
            "transient-similarity-threshold" => Ok(Self::TransientSimilarityThreshold(
                parse_nonnegative_f64(name, value)?,
            )),
            "transient-similarity-dark-threshold" => Ok(Self::TransientSimilarityDarkThreshold(
                parse_nonnegative_f64(name, value)?,
            )),
            "transient-similarity-mask-percent" => Ok(Self::TransientSimilarityMaskPercent(
                parse_percent(name, value)?,
            )),
            _ => Err(ParseDetectionOptionOverrideError::UnknownOption(
                name.to_string(),
            )),
        }
    }
}

impl FromStr for DetectionOptionOverride {
    type Err = ParseDetectionOptionOverrideError;

    #[inline]
    fn from_str(input: &str) -> Result<Self, Self::Err> {
        let input = input.trim().trim_start_matches("--");
        if input.is_empty() {
            return Err(ParseDetectionOptionOverrideError::Empty);
        }

        if let Some(name) = input.strip_prefix("no-") {
            return Self::from_name_value(name, "false");
        }

        let Some((name, value)) = input.split_once('=') else {
            return Self::from_name_value(input, "true");
        };
        if value.is_empty() {
            return Err(ParseDetectionOptionOverrideError::MissingValue(
                name.to_string(),
            ));
        }
        Self::from_name_value(name, value)
    }
}

/// Error returned when parsing a controlled scene detection option override.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseDetectionOptionOverrideError {
    /// The provided option string is empty.
    Empty,
    /// The option name is not in the controlled allowlist.
    UnknownOption(String),
    /// The option requires an explicit value but none was provided.
    MissingValue(String),
    /// The option value could not be parsed or is outside the accepted range.
    InvalidValue {
        /// The rejected option name.
        name: String,
        /// The rejected raw value.
        value: String,
        /// Short description of accepted values.
        expected: &'static str,
    },
}

impl fmt::Display for ParseDetectionOptionOverrideError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => write!(f, "empty av-scenechange option"),
            Self::UnknownOption(name) => write!(f, "unknown av-scenechange option '{name}'"),
            Self::MissingValue(name) => {
                write!(f, "av-scenechange option '{name}' requires a value")
            }
            Self::InvalidValue {
                name,
                value,
                expected,
            } => write!(
                f,
                "invalid value '{value}' for av-scenechange option '{name}' (expected {expected})"
            ),
        }
    }
}

impl Error for ParseDetectionOptionOverrideError {
}

impl DetectionOptions {
    /// Applies one controlled scene detection option override.
    #[inline]
    pub fn apply_override(&mut self, option: DetectionOptionOverride) {
        match option {
            DetectionOptionOverride::FlashDetection(enabled) => self.detect_flashes = enabled,
            DetectionOptionOverride::Lookahead(frames) => self.lookahead_distance = frames,
            DetectionOptionOverride::FastThresholdScale(scale) => {
                self.tuning.fast_threshold_scale = scale;
            }
            DetectionOptionOverride::ForwardSimilarity(enabled) => {
                self.tuning.forward_similarity.enabled = enabled;
            }
            DetectionOptionOverride::ForwardSimilarityFrames(frames) => {
                self.tuning.forward_similarity.frames = frames;
            }
            DetectionOptionOverride::ForwardSimilarityMinOffset(offset) => {
                self.tuning.forward_similarity.min_offset = offset;
            }
            DetectionOptionOverride::ForwardSimilarityThreshold(threshold) => {
                self.tuning.forward_similarity.threshold_8bit = threshold;
            }
            DetectionOptionOverride::ForwardSimilarityMaskPercent(percent) => {
                self.tuning.forward_similarity.mask_percent = percent;
            }
            DetectionOptionOverride::ForwardSimilarityRequireReturnCandidate(required) => {
                self.tuning.forward_similarity.require_return_candidate = required;
            }
            DetectionOptionOverride::ForwardSimilaritySuppressInside(enabled) => {
                self.tuning.forward_similarity.suppress_inside = enabled;
            }
            DetectionOptionOverride::TransientSimilarity(enabled) => {
                self.tuning.transient_similarity.enabled = enabled;
            }
            DetectionOptionOverride::TransientSimilarityFrames(frames) => {
                self.tuning.transient_similarity.frames = frames;
            }
            DetectionOptionOverride::TransientSimilarityThreshold(threshold) => {
                self.tuning.transient_similarity.threshold_8bit = threshold;
            }
            DetectionOptionOverride::TransientSimilarityDarkThreshold(threshold) => {
                self.tuning.transient_similarity.dark_threshold_8bit = threshold;
            }
            DetectionOptionOverride::TransientSimilarityMaskPercent(percent) => {
                self.tuning.transient_similarity.mask_percent = percent;
            }
        }
    }

    /// Applies controlled scene detection option overrides in order.
    #[inline]
    pub fn apply_overrides<I>(&mut self, options: I)
    where
        I: IntoIterator<Item = DetectionOptionOverride>,
    {
        for option in options {
            self.apply_override(option);
        }
    }
}

fn parse_bool(name: &str, value: &str) -> Result<bool, ParseDetectionOptionOverrideError> {
    match value {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        _ => Err(ParseDetectionOptionOverrideError::InvalidValue {
            name: name.to_string(),
            value: value.to_string(),
            expected: "true|false",
        }),
    }
}

fn parse_usize(name: &str, value: &str) -> Result<usize, ParseDetectionOptionOverrideError> {
    value
        .parse()
        .map_err(|_| ParseDetectionOptionOverrideError::InvalidValue {
            name: name.to_string(),
            value: value.to_string(),
            expected: "non-negative integer",
        })
}

fn parse_positive_usize(
    name: &str,
    value: &str,
) -> Result<usize, ParseDetectionOptionOverrideError> {
    let value = parse_usize(name, value)?;
    if value > 0 {
        Ok(value)
    } else {
        Err(ParseDetectionOptionOverrideError::InvalidValue {
            name: name.to_string(),
            value: value.to_string(),
            expected: "positive integer",
        })
    }
}

fn parse_f64(name: &str, value: &str) -> Result<f64, ParseDetectionOptionOverrideError> {
    let value =
        value
            .parse::<f64>()
            .map_err(|_| ParseDetectionOptionOverrideError::InvalidValue {
                name: name.to_string(),
                value: value.to_string(),
                expected: "finite number",
            })?;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(ParseDetectionOptionOverrideError::InvalidValue {
            name: name.to_string(),
            value: value.to_string(),
            expected: "finite number",
        })
    }
}

fn parse_nonnegative_f64(
    name: &str,
    value: &str,
) -> Result<f64, ParseDetectionOptionOverrideError> {
    let value = parse_f64(name, value)?;
    if value >= 0.0 {
        Ok(value)
    } else {
        Err(ParseDetectionOptionOverrideError::InvalidValue {
            name: name.to_string(),
            value: value.to_string(),
            expected: "non-negative finite number",
        })
    }
}

fn parse_percent(name: &str, value: &str) -> Result<f64, ParseDetectionOptionOverrideError> {
    let value = parse_f64(name, value)?;
    if (0.0..=0.95).contains(&value) {
        Ok(value)
    } else {
        Err(ParseDetectionOptionOverrideError::InvalidValue {
            name: name.to_string(),
            value: value.to_string(),
            expected: "number from 0.0 to 0.95",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{DetectionOptionOverride, FastThresholdScale};

    #[test]
    fn parses_controlled_options() {
        assert_eq!(
            "no-forward-similarity".parse(),
            Ok(DetectionOptionOverride::ForwardSimilarity(false))
        );
        assert_eq!(
            "--transient-similarity=false".parse(),
            Ok(DetectionOptionOverride::TransientSimilarity(false))
        );
        assert_eq!(
            "forward-similarity-frames=40".parse(),
            Ok(DetectionOptionOverride::ForwardSimilarityFrames(40))
        );
        assert_eq!(
            "fast-threshold-scale=sample-range".parse(),
            Ok(DetectionOptionOverride::FastThresholdScale(
                FastThresholdScale::SampleRange
            ))
        );
    }
}
