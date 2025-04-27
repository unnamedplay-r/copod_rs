//! # COPOD Rust Implementation
//!
//! A Rust implementation of the COPOD (Copula-Based Outlier Detection) algorithm[cite: 1].
//! Provides a scikit-learn like interface for fitting the model and predicting outlier scores.
//!
//! [1] Li, Z., Zhao, Y., Botta, N., Ionescu, C. and Hu, X. COPOD: Copula-Based Outlier Detection. IEEE International Conference on Data Mining (ICDM), 2020.

// Module declarations
mod copod_model;
mod internals;
mod types;

// Re-export public items
pub use copod_model::Copod;
pub use types::{CopodError, CopodVariant, DataMatrix, Result};

// Example Usage (can be put in tests or examples/main.rs)
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_workflow() {
        // 1. Create data (dummy data)
        let train_data: DataMatrix = vec![
            vec![1.0, 2.0],
            vec![1.1, 2.2],
            vec![0.9, 1.9],
            vec![1.0, 2.1],
            vec![10.0, 5.0], // Potential outlier
            vec![0.8, 1.8],
        ];

        let test_data: DataMatrix = vec![
            vec![1.2, 2.3],  // Normal point
            vec![15.0, 6.0], // Outlier
            vec![0.7, 1.7],  // Normal point
        ];

        // 2. Initialize model
        // Using SkewnessCorrected as it's generally the best performing variant
        let mut model = Copod::new(CopodVariant::SkewnessCorrected);

        // 3. Fit model
        let fit_result = model.fit(&train_data);
        assert!(fit_result.is_ok());
        println!("Fit successful.");

        // 4. Predict scores
        let predict_result = model.predict(&test_data);
        assert!(predict_result.is_ok());
        let scores = predict_result.unwrap();
        println!("Predicted scores: {:?}", scores);
        // Basic check: expect the score for the outlier test point to be higher (actual values depend on implementation)
        assert!(scores[1] > scores[0]);
        assert!(scores[1] > scores[2]);

        // 5. Explain an outlier
        let explain_result = model.explain(&test_data[1]); // Explain the outlier point
        assert!(explain_result.is_ok());
        let dimensional_scores = explain_result.unwrap();
        println!(
            "Dimensional scores for outlier {:?}: {:?}",
            test_data[1], dimensional_scores
        );
        // Expect high scores for dimensions contributing to outlierness (values depend on implementation)
    }

    #[test]
    fn test_not_fitted() {
        let model = Copod::new(CopodVariant::LeftTail);
        let test_data: DataMatrix = vec![vec![1.0, 2.0]];
        let predict_result = model.predict(&test_data);
        assert!(matches!(predict_result, Err(CopodError::NotFitted)));
        let explain_result = model.explain(&test_data[0]);
        assert!(matches!(explain_result, Err(CopodError::NotFitted)));
    }

    #[test]
    fn test_dimension_mismatch() {
        let train_data: DataMatrix = vec![vec![1.0, 2.0], vec![1.1, 2.1]];
        let test_data_wrong_dim: DataMatrix = vec![vec![1.0, 2.0, 3.0]];
        let mut model = Copod::new(CopodVariant::TwoTails);
        model.fit(&train_data).unwrap();

        let predict_result = model.predict(&test_data_wrong_dim);
        assert!(matches!(predict_result, Err(CopodError::DimensionMismatch)));

        let explain_result = model.explain(&test_data_wrong_dim[0]);
        assert!(matches!(explain_result, Err(CopodError::DimensionMismatch)));
    }

    #[test]
    fn test_fit_invalid_data_inconsistent_dimensions() {
        // An invalid matrix (inner vectors have different lengths)
        let invalid_matrix: DataMatrix = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0],
            vec![6.0, 7.0, 8.0, 9.0],
        ];
        let mut model = Copod::new(CopodVariant::SkewnessCorrected);
        let is_fit = model.fit(&invalid_matrix);
        assert!(matches!(is_fit, Err(CopodError::InvalidInput(_))));
    }

    #[test]
    fn test_fit_invalid_data_empty_outer_matrix() {
        // An empty outer vector
        let empty_matrix: DataMatrix = vec![];
        let mut model = Copod::new(CopodVariant::SkewnessCorrected);
        let is_fit = model.fit(&empty_matrix);
        assert!(matches!(is_fit, Err(CopodError::InvalidInput(_))));
    }

    #[test]
    fn test_fit_invalid_data_empty_inner_vectors() {
        // A matrix with empty inner vectors (all same length: 0)
        let empty_inner_matrix: DataMatrix = vec![vec![], vec![], vec![]];
        let mut model = Copod::new(CopodVariant::SkewnessCorrected);
        let is_fit = model.fit(&empty_inner_matrix);
        assert!(matches!(is_fit, Err(CopodError::InvalidInput(_))));
    }
}
