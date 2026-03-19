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

### How to use it
DivideIn10Folds.c makes a stratified partition of the dataset in training folds and testing folds. The dataset has to be in the working directory, and the way to execute the code through the command prompt is:
- .\DivideIn10Folds flare.dat 10
- 
If receives 2 parameters, the name of the dataset to partition and the number of folds (10 folds).

The execution returns:

- Classes.dat, a file with the class labels and their frequency
- Dataset1.dat, Dataset2.dat, ... Dataset10.dat: The training files, containing 90% of the instances that are used to mine the CARs. It follows the philosophy of a 10-fold cross validation.
- 1.dat, 2.dat, ... 10.dat: the corresponding testing files of each of the previous files, containing the remaining 10% of the partition.

SPAN-AC-Miner.c mines the positive and negative CARs. The user has to copy the files Dataset1.dat, Dataset2.dat, ... Dataset10.dat into the working directory, and also GenMax10Fold.bat, which generates the CARs per each training dataset. The lines of the .bat file have the following structure:
- .\SPAN-AC-Miner.exe Dataset1.dat 0.001 RulesDataset1.dat 0.01 4.0 0.25 0.1
- 
The parameters are:
- the input file (training dataset)
- the minimum support threshold 
- the output file with the mined CARs
- the minimum Netconf threshold for positive CARs
- the negative multiplier m   (to obtain the threshold for negative CARs)
- the quasi-exclusion threshold ε
- the WRAcc Filter α
After execution, the code return the 10 files with the positive and negative CARs, one per training dataset.

SPAN-AC.cpp is the final classifier. The user has to copy Classes.dat, the testing files 1.dat, 2.dat, ... 10.dat and the rule files RulesDataset1, RulesDataset2, ... RulesDataset10 into the working directory. The command to execute is:
- .\SPAN-AC Accuracies.dat
The parameter is the output file with the accuracies per fold and the average accuracy. The execution returns also:
- AnalisisInstancias.dat: contains informacion of every instance, like the assigned label class, the real class label, the CARs covering the instance per each class, and the count of correct classification until the current instance.
- stability_instance_log.csv: contains aditional info per each instance, like the best positive score, the second best positive score, if there is a near-tie situation that determines whether the covering negative CARs should be used or not to break a tie, etc.
- stability_fold_summary.csv: contains info of each testing fold, like number of instances, number of instances correctly classified and accuracy. 
