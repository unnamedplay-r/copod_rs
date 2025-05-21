#![allow(dead_code)] // Allow unused functions/structs for now
#![allow(unused_variables)] // Allow unused variables for now

use crate::internals;
use crate::types::DataProperties;
use crate::types::{CopodError, CopodVariant, DataMatrix, ECDF, FittedState, Result};

/// Column state vector
struct ColumnState {
    col_n: i64,
    left_tail_ecdf: ECDF,
    right_tail_ecdf: ECDF,
    skewness: Option<f64>,
}

/// The main COPOD model struct.
#[derive(Debug, Clone)]
pub struct Copod {
    variant: CopodVariant,
    fitted_state: Option<FittedState>,
    ecdf_approx: bool,
}

impl Copod {
    /// Creates a new COPOD model instance with specified hyperparameters.
    ///
    /// Args:
    ///     variant: The type of COPOD calculation to use (LeftTail, RightTail, TwoTails, SkewnessCorrected).
    pub fn new(variant: CopodVariant, ecdf_approx: bool) -> Self {
        Copod {
            variant,
            fitted_state: None,
            ecdf_approx, // TODO
        }
    }

    /// Fits the COPOD model to the training data.
    /// This involves calculating ECDFs and skewness for each dimension.
    ///
    /// Args:
    ///     data: The training data (rows are samples, columns are features).
    ///
    /// Returns:
    ///     Ok(()) if fitting is successful, Err(CopodError) otherwise.
    pub fn fit(&mut self, data: &DataMatrix) -> Result<()> {
        // TODO: Implement the fitting logic.
        // 1. Validate input data (e.g., not empty, consistent dimensions).
        // 2. Determine dimensions (d) and number of samples (n).
        // 3. For each dimension j from 0 to d-1:
        //    a. Extract the j-th column.
        //    b. Calculate and store the left-tail ECDF (using the column).
        //    c. Calculate and store the right-tail ECDF (using the *negated* column).
        //    d. If variant is SkewnessCorrected, calculate and store skewness.
        // 4. Store these calculated internals in `self.fitted_state`.

        // Validate input data
        // -------------------
        if data.is_empty() || data[0].is_empty() {
            return Err(CopodError::InvalidInput(
                "Input data cannot be empty".to_string(),
            ));
        }
        let n_samples = data.len();
        let n_dim = data.ndim();

        // confirm all data has a uniform internval vector state
        if data.is_uniform_inner_length() == false {
            return Err(CopodError::InvalidInput(
                "Input data has mismatched dimensions within the data".to_string(),
            ));
        }

        // Follows closely the logic from Algorithm 1.
        let mut col_state: Vec<ColumnState> = Vec::new();
        for col in 0..n_dim {
            let col_d: Vec<f64> = data.get_column_copy(col);

            // compute ecdf
            let left_tail_ecdf: ECDF = ECDF::fit(&col_d)?; // TODO: change this to a method

            /*
            we can do something more saucy here and just negate the unique count values instead of recomputing
            them over again. From there, we can just work with the cumulative sum of those negated features.
            it's also worth questioning if there's a direct relationship between the positive and negative
            cumulative sums of a set of values - it seems at first glance that there's a trivial relationship between
            them, on the same given set of data. */
            let neg_col_d = col_d.iter().map(|x| -x).collect::<Vec<f64>>();
            let right_tail_ecdf: ECDF = ECDF::fit(&neg_col_d)?;

            let skew_coef: Option<f64> = match self.variant {
                CopodVariant::SkewnessCorrected => Some(internals::calculate_skewness(&col_d)),
                _ => None,
            };

            col_state.push(ColumnState {
                col_n: col as i64,
                left_tail_ecdf,
                right_tail_ecdf,
                skewness: skew_coef,
            });
        }

        for col in 0..n_dim {
            let copula_state: &ColumnState = &col_state[col];

            copula_state
                .left_tail_ecdf
                .calculate_ecdf_value(column, value)
        }

        // Placeholder fitted state
        self.fitted_state = Some(FittedState {
            dimensions: n_dim,
            n_samples, /*, TODO: add ECDFs, skewness */
            contamination: 0.5,
        });

        println!(
            "Model fitted (stub implementation). Dimensions: {}, Samples: {}",
            n_dim, n_samples
        );
        Ok(())
    }

    /// Predicts outlier scores for new data points.
    /// Requires the model to be fitted first.
    ///
    /// Args:
    ///     data: The data points for which to predict outlier scores.
    ///
    /// Returns:
    ///     A vector of outlier scores (f64), one for each input data point, or an error.
    ///     Higher scores indicate a higher likelihood of being an outlier.
    pub fn predict(&self, data: &DataMatrix) -> Result<Vec<f64>> {
        // TODO: Implement the prediction logic.
        // 1. Check if the model is fitted (`self.fitted_state.is_some()`).
        //      If not, return Err(CopodError::NotFitted).
        // 2. Get the fitted state.
        // 3. For each data point `x_i` in `data`:
        //    a. Check for dimension consistency with the fitted model.
        //    b. Calculate left-tail empirical copula observations `U_i = [F_1(x_i1), ..., F_d(x_id)]`.
        //    c. Calculate right-tail empirical copula observations `V_i = [Fbar_1(x_i1), ..., Fbar_d(x_id)]`.
        //    d. Calculate the left-tail score `p_l = -sum(log(U_ij))`.
        //    e. Calculate the right-tail score `p_r = -sum(log(V_ij))`.
        //    f. If variant is SkewnessCorrected:
        //       i. Calculate skewness-corrected copula observations `W_i` based on fitted skewness.
        //       ii. Calculate the skewness-corrected score `p_s = -sum(log(W_ij))`.
        //    g. Determine the final outlier score `O(x_i)` based on the variant:
        //       - LeftTail: `p_l`
        //       - RightTail: `p_r`
        //       - TwoTails: `max(p_l, p_r)` (Equivalent to max{-sum log U, -sum log V})
        //       - SkewnessCorrected: `max(p_l, p_r, p_s)` (Algorithm 1 specifies max of all three)
        // 4. Return the vector of calculated scores.

        let fitted_state = self.fitted_state.as_ref().ok_or(CopodError::NotFitted)?;

        let mut scores = Vec::with_capacity(data.len());
        for point in data {
            if point.len() != fitted_state.dimensions {
                return Err(CopodError::DimensionMismatch);
            }
            // Placeholder score calculation
            let score = match self.variant {
                // Replace 0.0 with actual score calculation using internal functions
                CopodVariant::LeftTail => {
                    // let u_obs = internals::get_empirical_copula_observations(point, fitted_state, /* Left */)?;
                    // internals::calculate_neg_log_prob(&u_obs)
                    0.0
                }
                CopodVariant::RightTail => {
                    // let v_obs = internals::get_empirical_copula_observations(point, fitted_state, /* Right */)?;
                    // internals::calculate_neg_log_prob(&v_obs)
                    0.0
                }
                CopodVariant::TwoTails => {
                    // let u_obs = internals::get_empirical_copula_observations(point, fitted_state, /* Left */)?;
                    // let v_obs = internals::get_empirical_copula_observations(point, fitted_state, /* Right */)?;
                    // let p_l = internals::calculate_neg_log_prob(&u_obs);
                    // let p_r = internals::calculate_neg_log_prob(&v_obs);
                    // p_l.max(p_r)
                    0.0
                }
                CopodVariant::SkewnessCorrected => {
                    // let u_obs = internals::get_empirical_copula_observations(point, fitted_state, /* Left */)?;
                    // let v_obs = internals::get_empirical_copula_observations(point, fitted_state, /* Right */)?;
                    // let w_obs = internals::get_empirical_copula_observations(point, fitted_state, /* Skew Corrected */)?;
                    // let p_l = internals::calculate_neg_log_prob(&u_obs);
                    // let p_r = internals::calculate_neg_log_prob(&v_obs);
                    // let p_s = internals::calculate_neg_log_prob(&w_obs);
                    // p_l.max(p_r).max(p_s) // As per Algorithm 1
                    0.0
                }
            };
            scores.push(score);
        }

        Ok(scores)
    }

    /// Provides an explanation for the outlier score of a data point by returning dimensional contributions.
    /// Requires the model to be fitted first.
    /// This corresponds to the Dimensional Outlier Graph concept.
    ///
    /// Args:
    ///     point: The single data point (row vector) to explain.
    ///
    /// Returns:
    ///     A vector of scores, one for each dimension, indicating its contribution to outlierness, or an error.
    ///     The exact nature of the score depends on the internal implementation (e.g., max(-log(U_d), -log(V_d))).
    pub fn explain(&self, point: &[f64]) -> Result<Vec<f64>> {
        // TODO: Implement explanation logic.
        // 1. Check if fitted.
        // 2. Check dimension consistency.
        // 3. Use `internals::calculate_dimensional_scores` to get the contribution of each dimension.
        // 4. Return the dimensional scores.

        let fitted_state = self.fitted_state.as_ref().ok_or(CopodError::NotFitted)?;
        if point.len() != fitted_state.dimensions {
            return Err(CopodError::DimensionMismatch);
        }

        internals::calculate_dimensional_scores(point, fitted_state)
    }
}
