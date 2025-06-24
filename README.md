# Feature Attribution ≠ Causation: When SHAP Misleads

An interactive visualization demonstrating why SHAP values can be dangerously misleading when features are correlated, and how feature importance fundamentally differs from causal importance.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Version](https://img.shields.io/badge/version-1.0.0-green.svg)
![JavaScript](https://img.shields.io/badge/javascript-ES6-yellow.svg)

**Live Demo**: [View Visualization](https://dishant2009.github.io/shap-causation-viz)  
**Author**: Dishant ([GitHub](https://github.com/dishant2009) | [Email](mailto:digdarshidishant@gmail.com))

## Table of Contents

1. [Overview](#overview)
2. [The Critical Problem](#the-critical-problem)
3. [Key Concepts](#key-concepts)
4. [Features](#features)
5. [Installation & Usage](#installation--usage)
6. [Understanding the Visualization](#understanding-the-visualization)
7. [Technical Implementation](#technical-implementation)
8. [Mathematical Background](#mathematical-background)
9. [Real-World Scenarios](#real-world-scenarios)
10. [Why This Matters](#why-this-matters)
11. [Common Misconceptions](#common-misconceptions)
12. [Contributing](#contributing)
13. [References](#references)
14. [License](#license)

## Overview

This visualization reveals a fundamental misunderstanding in machine learning interpretability: **SHAP values show feature importance for predictions, NOT causal relationships**. In the presence of correlated features, SHAP can attribute high importance to features with zero causal effect while completely missing the true causes.

### Quick Example

Imagine a medical AI system:
- **SHAP says**: "High cholesterol is the most important feature (SHAP value: 0.8)"
- **Reality**: Cholesterol has ZERO causal effect; it's just correlated with age
- **Consequence**: Doctors prescribe unnecessary statins instead of addressing age-related issues

This tool demonstrates this critical distinction through interactive visualizations across multiple domains.

## The Critical Problem

### What People Think SHAP Does
- Identifies which features **cause** the outcome
- Shows what to change to affect predictions
- Reveals the "why" behind decisions

### What SHAP Actually Does
- Shows which features the **model uses** for predictions
- Measures **correlation-based** contribution
- Can be completely wrong about causation

### The Danger
When features are correlated (which is almost always), SHAP values become arbitrary and misleading:
- A feature with 0% causal effect can have 80% SHAP importance
- The true causal feature might show low importance
- Different models give different "explanations" for the same prediction

## Key Concepts

### SHAP (SHapley Additive exPlanations)
- Based on Shapley values from game theory
- Attributes model output to input features
- Satisfies mathematical properties: efficiency, symmetry, null player
- **Does NOT satisfy**: causality, correctness under correlation

### Causation vs Correlation
- **Correlation**: Statistical relationship between variables
- **Causation**: One variable directly affects another
- **Key Insight**: Models learn correlations, not causations

### Simpson's Paradox
A phenomenon where a trend appears in different groups but disappears or reverses when groups are combined. The visualization includes an interactive demonstration of this paradox.

## Features

### 1. Interactive Scenarios

#### Medical Diagnosis
- **Setup**: Predicting heart disease risk
- **Features**: Blood Pressure, Cholesterol, Age, Exercise
- **Truth**: Only Age and Exercise causally affect risk
- **Problem**: SHAP attributes importance to correlated Blood Pressure and Cholesterol

#### Loan Approval
- **Setup**: Predicting loan default
- **Features**: Income, Zip Code, Credit History, Education
- **Truth**: Only Income and Credit History matter
- **Problem**: SHAP highlights Zip Code (proxy for race) leading to discrimination

#### Hiring Decision
- **Setup**: Predicting job performance
- **Features**: Skills Test, University, Years Experience, References
- **Truth**: Only Skills Test and References predict performance
- **Problem**: SHAP emphasizes University/Experience, perpetuating bias

#### Simpson's Paradox Demo
- **Setup**: Medical treatment effectiveness
- **Shows**: Treatment appears harmful overall but helps in every subgroup
- **Lesson**: Aggregated SHAP values can be completely misleading

### 2. Real-Time Correlation Adjustment

- **Slider Range**: 0 to 0.95 correlation
- **Effect**: Watch SHAP attribution error grow from 0% to 80%+
- **Insight**: Even small correlations (0.3) cause significant errors

### 3. Visual Components

#### Side-by-Side Feature Importance
- **Left Panel**: SHAP values (what practitioners see)
- **Right Panel**: True causal effects (reality)
- **Color Coding**: 
  - Green: Correct attribution
  - Red: False attribution
  - Bold: True causal features

#### Scatter Plot Visualization
- Shows correlation between features
- Interactive points with detailed tooltips
- Color indicates outcome (blue: positive, red: negative)

#### Causal Structure Diagram
- Displays true causal relationships
- Shows correlation structure
- Updates based on selected scenario

#### Model Comparison
- Two models with identical accuracy
- Completely different SHAP explanations
- Demonstrates arbitrariness of explanations

### 4. Metrics Dashboard

- **Feature Correlation**: Current correlation strength
- **SHAP Attribution Error**: Percentage of importance wrongly attributed
- **Correct Causal Attribution**: How much SHAP gets right
- **Decision Risk**: High/Medium/Low based on error rate

## Installation & Usage

### Option 1: Direct Browser Usage
Simply open the HTML file in any modern browser. No installation required.

```bash
# Clone the repository
git clone https://github.com/dishant2009/shap-causation-viz.git

# Open in browser
open shap-causation-viz.html
```

### Option 2: Local Server (Recommended)
```bash
# Using Python
python -m http.server 8000

# Using Node.js
npx http-server

# Navigate to
http://localhost:8000/shap-causation-viz.html
```

### Requirements
- Modern web browser (Chrome, Firefox, Safari, Edge)
- JavaScript enabled
- No external dependencies (pure vanilla JS)

## Understanding the Visualization

### Reading the Feature Bars

#### SHAP Values (Left Panel)
- **Bar Length**: Magnitude of SHAP value
- **Bar Direction**: Positive (right) or negative (left) effect
- **Red Values**: False importance (non-causal features with high SHAP)
- **What it shows**: How the model distributes credit

#### True Causal Effects (Right Panel)
- **Bar Length**: Actual causal effect size
- **Bar Direction**: Increases (right) or decreases (left) outcome
- **Green Values**: True causal features
- **Zero Bars**: Features with no causal effect

### Interpreting the Scatter Plot

- **X-axis**: First feature in scenario
- **Y-axis**: Second feature in scenario
- **Point Color**: Blue = positive outcome, Red = negative outcome
- **Point Distribution**: Shows correlation pattern
- **Hover**: Reveals all feature values for that data point

### Understanding Correlation Impact

As you increase correlation:
1. **0-0.2**: Minimal SHAP distortion
2. **0.3-0.5**: Significant false attributions begin
3. **0.6-0.8**: Majority of importance wrongly attributed
4. **0.9+**: SHAP becomes essentially random

### The Simpson's Paradox Demonstration

Shows four statistics boxes:
1. **Overall**: Treatment appears harmful
2. **Mild Cases**: Treatment helps
3. **Severe Cases**: Treatment helps
4. **Paradox Explanation**: Why aggregation misleads

## Technical Implementation

### Architecture

```
shap-causation-viz/
├── index.html          # Single-file visualization
├── README.md          # This documentation
└── LICENSE            # MIT license
```

### Core Components

#### Data Generation
```javascript
function generateData() {
    // 1. Generate true causal features
    // 2. Add correlated features based on correlation strength
    // 3. Calculate outcome using ONLY causal features
    // 4. Create 200 data points per scenario
}
```

#### SHAP Simulation
```javascript
// SHAP incorrectly attributes importance to correlated features
let shapValue = causalEffect;
if (isCorrelated) {
    const sourceEffect = scenario.causal[sourceFeature];
    shapValue = sourceEffect * correlationStrength;
}
```

#### Visualization Updates
```javascript
function updateVisualization() {
    updateMetrics();        // Dashboard values
    updateFeatureBars();    // SHAP vs causal bars
    updateScatterPlot();    // Correlation visualization
    updateCausalDiagram();  // True structure
    updateModelComparison(); // Different explanations
}
```

### Key Design Decisions

1. **No External Libraries**: Pure JavaScript for maximum compatibility
2. **Real-Time Updates**: All changes reflected immediately
3. **Scenario-Based**: Real-world examples instead of abstract data
4. **Visual First**: Complex concepts made intuitive through visualization

## Mathematical Background

### SHAP Values

SHAP values are based on Shapley values from cooperative game theory:

```
φᵢ = Σ[S⊆F\{i}] [|S|!(|F|-|S|-1)! / |F|!] [f(S∪{i}) - f(S)]
```

Where:
- φᵢ = SHAP value for feature i
- F = Set of all features
- S = Subset of features
- f = Model prediction function

### The Correlation Problem

When features X and Y are correlated:
- Cov(X,Y) ≠ 0
- Model can use either X or Y for prediction
- SHAP arbitrarily distributes importance between them
- True causal effect could be entirely in X, Y, or neither

### Simpson's Paradox

For groups A and B:
- P(Outcome|Treatment, A) > P(Outcome|Control, A)
- P(Outcome|Treatment, B) > P(Outcome|Control, B)
- But: P(Outcome|Treatment) < P(Outcome|Control)

This occurs when group membership correlates with both treatment and outcome.

## Real-World Scenarios

### Medical AI Failures

**Scenario**: Heart disease prediction model

**What Happens**:
1. Model trained on data where age correlates with cholesterol
2. SHAP says "cholesterol is key factor"
3. Doctors prescribe statins to young patients with high cholesterol
4. Real risk (age) is ignored

**Impact**: Unnecessary medication, missed interventions

### Discriminatory Lending

**Scenario**: Loan approval system

**What Happens**:
1. Zip code correlates with income and race
2. SHAP highlights zip code as important
3. Model appears to use "geographic data"
4. Actually performing racial discrimination

**Impact**: Illegal discrimination, regulatory violations

### Biased Hiring

**Scenario**: Resume screening AI

**What Happens**:
1. University prestige correlates with actual skills
2. SHAP shows university as key factor
3. HR thinks education matters most
4. Skilled candidates from other schools rejected

**Impact**: Missed talent, perpetuated inequality

### Policy Mistakes

**Scenario**: Education intervention planning

**What Happens**:
1. Tablet usage correlates with family income
2. SHAP shows tablets improve test scores
3. Government provides tablets to all students
4. Real cause (tutoring, resources) not addressed

**Impact**: Wasted resources, continued education gaps

## Why This Matters

### For Data Scientists
- SHAP ≠ causal inference
- Need separate causal analysis tools
- Must understand model limitations
- Correlation structure affects all interpretations

### For Decision Makers
- "AI explanations" can be completely wrong
- High stakes decisions need causal analysis
- Regulatory compliance requires true understanding
- Black box models + SHAP ≠ interpretability

### For Society
- AI systems can perpetuate discrimination
- Medical decisions based on correlations can harm
- Policy interventions may target wrong factors
- Trust in AI requires honest limitations

## Common Misconceptions

### Misconception 1: "SHAP tells me what to change"
**Reality**: SHAP shows correlations. Changing a correlated feature may have no effect.

### Misconception 2: "Higher SHAP = More important feature"
**Reality**: High SHAP can occur for completely non-causal features.

### Misconception 3: "SHAP explanations are unique"
**Reality**: Different models give different SHAP explanations for identical predictions.

### Misconception 4: "Complex models + SHAP = Interpretable AI"
**Reality**: Adding explanations to black boxes doesn't make them trustworthy.

### Misconception 5: "SHAP is always better than simpler methods"
**Reality**: For causal understanding, controlled experiments or causal models are essential.

## Contributing

Contributions are welcome! Areas for improvement:

1. **Additional Scenarios**: Healthcare, criminal justice, climate modeling
2. **Causal Methods**: Add demonstrations of proper causal inference
3. **Advanced Features**: Multi-feature correlations, time series
4. **Educational Content**: Tutorials, exercises, quizzes
5. **Accessibility**: Screen reader support, keyboard navigation

### Development Guidelines

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Code Style
- Clear variable names
- Extensive comments
- Consistent indentation (4 spaces)
- JSDoc for functions

## References

### Core Papers

1. **Lundberg & Lee (2017)** - "A Unified Approach to Interpreting Model Predictions"
   - Introduced SHAP
   - [Paper](https://arxiv.org/abs/1705.07874)

2. **Pearl (2009)** - "Causality: Models, Reasoning, and Inference"
   - Foundation of causal inference
   - Essential for understanding the distinction

3. **Janzing et al. (2020)** - "Feature relevance quantification in explainable AI"
   - Discusses SHAP limitations
   - [Paper](https://arxiv.org/abs/1910.13413)

4. **Kumar et al. (2020)** - "Problems with Shapley-value-based explanations"
   - Formal analysis of SHAP issues
   - [Paper](https://arxiv.org/abs/2002.11097)

### Related Resources

- [Causal Inference: The Mixtape](https://mixtape.scunning.com/)
- [Elements of Causal Inference](https://mitpress.mit.edu/books/elements-causal-inference)
- [The Book of Why](https://www.basicbooks.com/titles/judea-pearl/the-book-of-why/9780465097609/)

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contact

**Author**: Dishant  
**GitHub**: [@dishant2009](https://github.com/dishant2009)  
**Email**: digdarshidishant@gmail.com

---

**Remember**: Feature importance is not causal importance. When making high-stakes decisions, understanding this distinction can be the difference between helping and harming.
