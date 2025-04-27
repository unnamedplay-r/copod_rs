#![allow(dead_code)] // Allow unused functions for now
#![allow(unused_variables)] // Allow unused variables for now

use crate::types::{BTreeFloat, CopodError, FittedState, Result, ECDF};
use std::collections::BTreeMap;

fn count_unique_floats_as_two_vecs(numbers: &[f64]) -> (Vec<f64>, Vec<usize>) {
    let mut counts: BTreeMap<BTreeFloat, usize> = BTreeMap::new();

    for num in numbers {
        let comparable_num = BTreeFloat(*num);
        *counts.entry(comparable_num).or_insert(0) += 1;
    }

    // The iterator yields (f64, usize) after the map. unzip() splits this.
    counts.into_iter()
        .map(|(k, v)| (k.0, v)) // Transform (BTreeFloat, usize) to (f64, usize)
        .unzip()               // Collects into (Vec<f64>, Vec<usize>)
}


/// Internal function to calculate the Empirical Cumulative Distribution Function (ECDF)
/// for a single dimension (column) of the data.
/// This corresponds to Equation 5 in the paper.
///
/// Args:
///     column: A slice representing a single feature/dimension.
///     value: The value at which to evaluate the ECDF.
///
/// Returns:
///     The ECDF value P(X <= value).
pub(crate) fn calculate_ecdf_value(column: &[f64], value: f64) -> f64 {
    // TODO: Implement ECDF calculation: count(x_i <= value) / n
    let n = column.len() as f64;
    if n == 0.0 {
        return 0.0;
    }
    let count = column.iter().filter(|&&x| x <= value).count() as f64;
    count / n
}

/// Internal function to calculate the ECDF for an entire column.
/// This might return a representation useful for quick lookups.
///
/// Args:
///     column: A slice representing a single feature/dimension.
///
/// Returns:
///     A representation of the ECDF (e.g., sorted values and ranks).
pub(crate) fn fit_ecdf(column: &[f64]) -> Result<ECDF> {
    // TODO: Implement logic to efficiently represent the ECDF for a column.
    // Taking inspiration from both:
    // [1] https://www.statsmodels.org/stable/_modules/statsmodels/distributions/empirical_distribution.html#ECDF
    // [2] https://github.com/scipy/scipy/blob/main/scipy/stats/_survival.py#L18
    let (unique_values, counts) = count_unique_floats_as_two_vecs(column);
    Ok(())
}

/// Internal function to calculate the skewness of a single dimension (column).
/// Corresponds to Equation 11 in the paper.
///
/// Args:
///     column: A slice representing a single feature/dimension.
///
/// Returns:
///     The calculated skewness value. Returns 0.0 if calculation isn't possible (e.g., n < 3 or std dev is 0).
pub(crate) fn calculate_skewness(column: &[f64]) -> f64 {
    // TODO: Implement skewness calculation.
    let n = column.len();
    if n < 3 {
        return 0.0;
    } // Skewness requires at least 3 points

    let mean = column.iter().sum::<f64>() / (n as f64);
    let variance = column.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / ((n - 1) as f64); // Use sample variance n-1

    if variance == 0.0 {
        return 0.0;
    } // Avoid division by zero if all points are the same
    let std_dev = variance.sqrt();

    let m3 = column.iter().map(|&x| (x - mean).powi(3)).sum::<f64>() / (n as f64); // Third central moment (biased estimator is fine here per paper)
    let skewness = m3 / std_dev.powi(3);

    skewness
}

/// Internal function to calculate the empirical copula observations for a single data point.
/// Corresponds to Equation 6 in the paper, potentially using left, right, or skewness-corrected ECDFs.
///
/// Args:
///     point: A slice representing a single data point (observation).
///     fitted_state: The fitted state containing ECDFs and skewness.
///
/// Returns:
///     A vector of copula observations [U_1, U_2, ..., U_d].
pub(crate) fn get_empirical_copula_observations(
    point: &[f64],
    fitted_state: &FittedState,
    // Add parameters to specify which ECDF type (left, right, skew-corrected) to use
) -> Result<Vec<f64>> {
    // TODO: Implement logic to get U_j = F_j(x_j) for each dimension j.
    // This requires looking up values in the pre-calculated ECDFs stored in fitted_state.
    // Need to handle left-tail (using X) and right-tail (using -X) ECDFs.
    // For skewness correction, select U_j from left or right based on fitted_state.skewness[j].
    if point.len() != fitted_state.dimensions {
        return Err(CopodError::DimensionMismatch);
    }
    // Placeholder implementation
    let observations = point.iter().map(|_| 0.5).collect(); // Replace with actual calculation
    Ok(observations)
}

/// Internal function to calculate the negative log probability score for a point based on its copula observations.
/// Corresponds to Equation 10 and subsequent steps in Algorithm 1.
///
/// Args:
///     copula_observations: The vector [U_1, ..., U_d] for the point.
///
/// Returns:
///     The calculated score: -sum(log(U_j)). Handles log(0) by returning positive infinity or a large number.
pub(crate) fn calculate_neg_log_prob(copula_observations: &[f64]) -> f64 {
    // TODO: Implement the negative log probability sum.
    let mut score = 0.0;
    for &u in copula_observations {
        if u <= 0.0 {
            // Handle edge case: log(0) is -infinity, -log(0) is +infinity.
            // Return positive infinity or a very large number to indicate extreme outlierness.
            return f64::INFINITY;
        }
        score -= u.ln(); // Natural logarithm
    }
    score
}

/// Internal function to calculate the dimensional outlier scores for explanation.
/// Based on Section II-E, calculating max{-log(U_d), -log(V_d), -log(W_d)} per dimension.
///
/// Args:
///     point: A slice representing a single data point (observation).
///     fitted_state: The fitted state containing ECDFs and skewness.
///
/// Returns:
///     A vector of dimensional outlier scores.
pub(crate) fn calculate_dimensional_scores(
    point: &[f64],
    fitted_state: &FittedState,
) -> Result<Vec<f64>> {
    if point.len() != fitted_state.dimensions {
        return Err(CopodError::DimensionMismatch);
    }
    // TODO: Implement dimensional score calculation.
    // For each dimension d:
    // 1. Get the left-tail copula observation U_d = F_d(x_d)
    // 2. Get the right-tail copula observation V_d = F_bar_d(x_d) (using ECDF of -X)
    // 3. Get the skewness-corrected observation W_d (U_d if skew < 0 else V_d)
    // 4. Calculate neg_log_u = -log(U_d), neg_log_v = -log(V_d), neg_log_w = -log(W_d) (handle 0s)
    // 5. Dimensional score O_d = max(neg_log_u, neg_log_v, neg_log_w) - although paper seems to simplify O_d(xi)=max{log(Ud,i)., log(Vd,i), log(Wd,i)} - check eq 96
    //    Let's use the negative log for consistency with the final score: O_d = max(-ln(U_d), -ln(V_d))  implies using the raw max, but algorithm 1  implies neg log. Let's stick to neg log for consistency.
    //    Re-reading Sec II-E, it seems to suggest O_d(xi)=max{log(Ud,i), log(Vd,i), log(Wd,i)}, which differs from the final score calculation using negative logs.
    //    However, the *intent* is to show contribution to the final score. Let's calculate -log(U_d), -log(V_d), -log(W_d) for interpretability.
    //    Final Score O(xi) = max{pl, pr, ps} where pl = -sum(log(Uj)), pr = -sum(log(Vj)), ps = -sum(log(Wj)) 
    //    Let's provide the individual terms: [-log(U_d), -log(V_d), -log(W_d)] for each dimension d. This provides maximum explanation power.

    let mut dimensional_scores: Vec<Vec<f64>> = Vec::with_capacity(fitted_state.dimensions);

    // Placeholder: Need actual ECDFs in fitted_state
    for d in 0..fitted_state.dimensions {
        // Placeholder values - replace with actual calculations using ECDFs
        let u_d = 0.5; // Replace with F_d(point[d])
        let v_d = 0.5; // Replace with Fbar_d(point[d]) calculated from ECDF of -X
        // let w_d = if fitted_state.skewness[d] < 0.0 { u_d } else { v_d }; // Calculate W_d

        let neg_log_u = if u_d <= 0.0 { f64::INFINITY } else { -u_d.ln() };
        let neg_log_v = if v_d <= 0.0 { f64::INFINITY } else { -v_d.ln() };
        // let neg_log_w = if w_d <= 0.0 { f64::INFINITY } else { -w_d.ln() }; // Can derive from u/v

        dimensional_scores.push(vec![neg_log_u, neg_log_v]); // Provide contributions from left and right tails
    }

    // For simplicity in this stub, let's just return a placeholder vector of size d
    let placeholder_scores = vec![0.0; fitted_state.dimensions];
    Ok(placeholder_scores) // TODO: Return the actual scores calculated above.
}
