import numpy as np

class TrafficAnomalyBaseline:
    """
        Baseline model for critical threat detection based on extreme traffic.

        This class implements a simple statistical anomaly detector using the 
        Interquartile Range (IQR) method. It identifies data points that exceed 
        a calculated threshold based on the distribution of training data.

        The IQR method is a standard statistical approach for outlier detection 
        that is robust to extreme values and does not assume normal distribution.

        Attributes
        ----------
        -k : float
            Multiplier for the IQR to define the upper threshold. Standard value 
            is 1.5 (used in boxplots). Higher values (e.g., 3.0) are more 
            conservative and detect fewer anomalies.
        -features : list of str
            List of feature names to analyze for outlier detection.
        -thresholds_ : dict
            Dictionary storing Q1, Q3, IQR, and threshold values for each feature.
            Populated after calling fit().

        Methods
        -------
        -fit(df):
            Calculate thresholds based on training data.
        -predict(df):
            Classify samples as critical (1) or normal (0).
        -get_anomaly_score(df):
            Calculate continuous anomaly scores for ranking.

        Examples
        --------
        >>> model = TrafficAnomalyBaseline(k=1.5, features=['bytes_in', 'bytes_out'])
        >>> model.fit(train_data)
        >>> predictions = model.predict(test_data)
        >>> scores = model.get_anomaly_score(test_data)
    """
    def __init__(self, k=1.5, features=['bytes_in', 'bytes_out']):
        """
            Initialize the baseline anomaly detector.
            
            Parameters
            ----------
            -k : float, default=1.5
                IQR multiplier for threshold calculation. Standard values:
                - 1.5: Standard (detects ~7% as outliers in normal distributions)
                - 3.0: Conservative (detects ~0.3% as outliers)
            -features : list of str, default=['bytes_in', 'bytes_out']
                Feature names to use for anomaly detection. The detector will 
                flag a sample as anomalous if ANY feature exceeds its threshold.
        """
        self.k = k
        self.features = features
        self.thresholds_ = {}
        
        
    def fit(self, df):
        """
            Learn thresholds from training data using the IQR method.
            
            For each feature, calculates:
            - Q1 (25th percentile)
            - Q3 (75th percentile)
            - IQR = Q3 - Q1
            - Upper threshold = Q3 + k * IQR
            
            Any value above the upper threshold is considered an outlier.
            
            Parameters
            ----------
            -df : pandas.DataFrame
                Training data containing the specified features.
            
            Returns
            -------
            -self : TrafficAnomalyBaseline
                Returns self to allow method chaining (e.g., model.fit(train).predict(test))
            
            Notes
            -----
            The IQR method is robust to outliers in the training data itself,
            as it uses percentiles rather than mean/standard deviation.
        """
        print(f"IQR K = {self.k}")
        print(f"Features: {', '.join(self.features)}")
        print("\n" + "-" * 80)
        
        for feature in self.features:
            Q1 = df[feature].quantile(0.25)
            Q3 = df[feature].quantile(0.75)
            IQR = Q3 - Q1
            upper_threshold = Q3 + self.k * IQR
            
            self.thresholds_[feature] = {
                'Q1': Q1,
                'Q3': Q3,
                'IQR': IQR,
                'threshold': upper_threshold
            }
            
            print(f"\n{feature}:")
            print(f"  Q1:        {Q1:>15,.0f}")
            print(f"  Q3:        {Q3:>15,.0f}")
            print(f"  IQR:       {IQR:>15,.0f}")
            print(f"  Umbral:    {upper_threshold:>15,.0f} (Q3 + {self.k} * IQR)")
            
        return self
    
    
    def predict(self, df):
        """
            Predict whether samples are critical (1) or normal (0).
            
            A sample is classified as critical if ANY of its features exceed 
            the corresponding threshold calculated during fit().
            
            Parameters
            ----------
            -df : pandas.DataFrame
                Data to classify, containing the same features used in fit().
            
            Returns
            -------
            -predictions : numpy.ndarray of shape (n_samples,)
                Binary predictions where:
                    -1 = Critical anomaly (at least one feature exceeds threshold)
                    -0 = Normal (all features below thresholds)
            
            Notes
            -----
            Uses logical OR across features: exceeding threshold in ANY feature
            triggers a critical classification.
        """
        predictions = np.zeros(len(df), dtype=int)
        for feature in self.features:
            threshold = self.thresholds_[feature]['threshold']
            predictions |= (df[feature] > threshold).astype(int)
        return predictions
    
    
    def get_anomaly_score(self, df):
        """
            Calculate continuous anomaly scores for prioritization.
            
            The score represents how much a sample violates the thresholds,
            normalized by IQR for interpretability. Higher scores indicate
            more extreme anomalies.
            
            Parameters
            ----------
            df : pandas.DataFrame
                Data to score, containing the same features used in fit().
            
            Returns
            -------
            -scores : numpy.ndarray of shape (n_samples,)
                -Anomaly scores where:
                    0 = Below all thresholds (normal)
                    > 0 = Exceeds at least one threshold (higher = more anomalous)
                    Score = max((feature_value - threshold) / IQR) across features
            
            Notes
            -----
            The score is useful for:
            - Ranking anomalies by severity
            - Setting dynamic alert thresholds (e.g., alert if score > 5.0)
            - Understanding which samples are borderline vs extreme
            
            Examples
            --------
            >>> scores = model.get_anomaly_score(test_data)
            >>> top_threats = test_data[scores > 5.0]  # Most critical threats
        """
        scores = np.zeros(len(df))
        for feature in self.features:
            threshold = self.thresholds_[feature]['threshold']
            iqr = self.thresholds_[feature]['IQR']
            violation = (df[feature] - threshold) / iqr
            violation = np.maximum(violation, 0)
            scores = np.maximum(scores, violation)
        return scores