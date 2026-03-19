# SPAN-AC
SPAN-AC is a stable associative classification framework that integrates positive and negative class association rules using a shrinkage-based scoring mechanism and a controlled, ambiguity-driven tie-breaking strategy.

## SPAN-AC: Stable Associative Classification with Controlled Negative CARs

SPAN-AC is an associative classification framework designed to improve
decision stability and reliability in rule-based classifiers.

The method combines positive and negative class association rules (CARs)
within a unified mining and classification process. Positive rules provide
supporting evidence for a class, while negative rules represent
quasi-exclusion and are used selectively to resolve ambiguous cases.

A key component of SPAN-AC is a shrinkage-based scoring mechanism that
stabilizes the aggregation of positive rule evidence. This prevents
overconfident predictions when only a small number of rules cover an
instance, which is a common issue in traditional associative classifiers.

Negative CARs are not used indiscriminately. Instead, they are activated
only in near-tie situations where multiple classes receive similar levels
of positive support. This conservative strategy allows negative evidence
to act as a precise conflict-resolution mechanism without introducing
instability or over-penalization.

The framework also incorporates a maximal-first mining strategy that
reduces redundancy and improves structural stability across different
training samples.

### Key characteristics

- Stable rule-based classification under noisy and imbalanced data
- Controlled integration of negative class association rules
- Shrinkage-based aggregation of positive evidence
- Deterministic and interpretable decision process
- Reduced rule redundancy via maximal pattern mining

SPAN-AC is particularly suitable for applications where interpretability,
consistency, and robustness are essential, such as healthcare, security,
and decision support systems.
