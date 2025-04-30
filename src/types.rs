#![allow(dead_code)] // Allow unused variants for now

// use std::collections::BTreeMap;
use std::cmp::Ordering;

/// Defines the different ways to calculate tail probabilities in COPOD.
/// Based on the paper's discussion of left, right, two-tail, and skewness-corrected approaches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CopodVariant {
    LeftTail,
    RightTail,
    TwoTails,          // Represents using the max of left and right tail probabilitie
    SkewnessCorrected, // Selects tail based on skewness of each dimension
}

// ---------

/// Represents the fitted internal state of the COPOD model.
/// This might hold calculated ECDFs, skewness values, etc.
#[derive(Debug, Clone)]
pub(crate) struct FittedState {
    // Placeholder: dimensions of the fitted data
    pub(crate) dimensions: usize,
    // Placeholder: number of samples used for fitting
    pub(crate) n_samples: usize,
    // The amount of contamination of the data set, i.e. the proportion of outliers in the data set. Used when fitting to
    // define the threshold on the decision function.
    pub(crate) contamination: f32,
    // TODO: Add fields to store calculated ECDFs (left/right) and skewness values per dimension.
    // These will be calculated during the `fit` process.
    // Example: pub(crate) left_ecdfs: Vec<SomeEcdfRepresentation>,
    // Example: pub(crate) right_ecdfs: Vec<SomeEcdfRepresentation>,
    // Example: pub(crate) skewness: Vec<f64>,
}

// ---------

/// Custom error types for the library (optional but recommended).
#[derive(Debug)]
pub enum CopodError {
    NotFitted,
    DimensionMismatch,
    InvalidInput(String),
    // TODO: Add other potential error cases
}

// Implement std::error::Error and std::fmt::Display for CopodError
impl std::fmt::Display for CopodError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CopodError::NotFitted => write!(f, "Model has not been fitted yet."),
            CopodError::DimensionMismatch => write!(f, "Input data has incorrect dimensions."),
            CopodError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
        }
    }
}

impl std::error::Error for CopodError {}

// Define a standard Result type for the library
pub type Result<T> = std::result::Result<T, CopodError>;

// -----------

// Note: We represent data as Vec<Vec<f64>>, where outer Vec is rows (samples)
// and inner Vec is columns (features/dimensions).
pub type DataMatrix = Vec<Vec<f64>>;

pub trait DataProperties {
    fn ndim(&self) -> usize;
    fn is_uniform_inner_length(&self) -> bool;
    fn get_column_copy(&self, d: usize) -> Vec<f64>;
}

impl DataProperties for DataMatrix {
    fn ndim(&self) -> usize {
        self[0].len()
    }

    fn is_uniform_inner_length(&self) -> bool {
        // Get the expected length from the first inner vector.
        let expected_len = self[0].len();

        // Iterate through the rest of the inner vectors and compare their lengths
        for inner_vec in self.iter().skip(1) {
            if inner_vec.len() != expected_len {
                // Found an inner vector with a different length
                return false;
            }
        }

        // If we've iterated through all inner vectors and haven't found a
        // different length, then all inner vectors have the same length.
        true
    }

    fn get_column_copy(&self, d: usize) -> Vec<f64> {
        // get the dth item from each vector, then flatten them into a new vector
        self.iter()
            .filter_map(|v| v.get(d))
            .copied()
            .collect::<Vec<f64>>()
    }
}

// ---------

/// We need the traits for equality on the float, but unfortunately, we can't modify
/// the existing float in rust (i think?), and I definitely should be using an existing, well tested
/// library for this, but i'm willing to explore how to modify floats to use a BTree here given that we're
/// strictly working with floats.
#[derive(Debug, Clone, Copy)] // Derive useful traits
pub(crate) struct BTreeFloat(pub f64);

// Implement PartialEq and Eq using total_cmp
impl PartialEq for BTreeFloat {
    fn eq(&self, other: &Self) -> bool {
        self.0.total_cmp(&other.0).is_eq()
    }
}
impl Eq for BTreeFloat {} // Eq is a marker trait indicating PartialEq provides an equivalence relation

// Implement PartialOrd and Ord using total_cmp
impl PartialOrd for BTreeFloat {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.0.total_cmp(&other.0)) // total_cmp never returns None
    }
}
impl Ord for BTreeFloat {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

// ---------

/// Empirical Cumulative Distribution Function, with inspiration from SciPy
pub(crate) struct ECDF {
    pub counts: Vec<u64>,
    pub quantiles: Vec<f64>,
}

impl ECDF {
    fn fit(data: Vec<f64>) -> ECDF {
        todo!();

        // let mut uniq_n_counts = BTreeMap::new();
        // counts = data
        // ECDF {}
    }
}
