# T-Learner (Two-Model Learner): Mathematical Foundation

## Core Concept

The **T-Learner** estimates heterogeneous treatment effects by training **two separate models**: one for the treatment group and one for the control group.

## Mathematical Framework

**Goal**: Estimate the Conditional Average Treatment Effect (CATE):

$$\tau(X) = E[Y(1) - Y(0) | X]$$

Where:
- $Y(1)$ = potential outcome if treated
- $Y(0)$ = potential outcome if not treated  
- $X$ = individual characteristics (covariates)

**T-Learner Approach:**

1. **Split the data** by treatment status:
   - Control group: $(X_i, Y_i)$ where $T_i = 0$
   - Treatment group: $(X_i, Y_i)$ where $T_i = 1$

2. **Train two separate models**:
   - $\mu_0(X) = E[Y | T=0, X]$ (control model)
   - $\mu_1(X) = E[Y | T=1, X]$ (treatment model)

3. **Estimate individual treatment effect**:
   $$\hat{\tau}(X) = \mu_1(X) - \mu_0(X)$$

## Causal DAG

```
        X (Covariates)
       / \
      /   \
     ↓     ↓
    T → Y (Outcome)
```

**Key Relationships:**
- $X$ → $T$: Covariates may influence treatment assignment (confounding)
- $X$ → $Y$: Covariates affect outcomes
- $T$ → $Y$: Treatment affects outcome (causal effect we want)

## Critical Assumptions

### 1. Unconfoundedness (Ignorability)

$$Y(0), Y(1) \perp T \mid X$$

**Meaning**: Once we condition on observed covariates $X$, treatment assignment is "as good as random"
- All confounders are **observed and included** in $X$
- No unmeasured confounding
- **This is the strongest and most critical assumption**

### 2. Overlap (Common Support)

$$0 < P(T=1|X) < 1$$

**Meaning**: Every individual has some chance of being in both treatment and control groups
- Can't have groups that are **always** treated or **never** treated
- Need comparable individuals in both groups

### 3. SUTVA (Stable Unit Treatment Value Assumption)

- **No interference between units**: My treatment doesn't affect your outcome
- **One version of treatment**: Treatment is well-defined and consistent

### 4. Consistency

$$Y = T \cdot Y(1) + (1-T) \cdot Y(0)$$

**Meaning**: The observed outcome equals the potential outcome under the actual treatment received

## Advantages of T-Learner

✅ **Flexibility**: Each model can use different algorithms  
✅ **Captures heterogeneity**: Different features can matter differently for treated vs. control  
✅ **Interpretable**: Clear separation between treatment/control models  
✅ **Non-parametric**: No need to specify functional form of treatment effect

## Disadvantages

❌ **Data splitting**: Each model sees only half the data (less efficient)  
❌ **Regularization issues**: Two separate models may overfit differently  
❌ **Ignores treatment propensity**: Doesn't account for $P(T=1|X)$  
❌ **Imbalanced groups**: Performs poorly when treatment/control groups are very different sizes

## Application to Reunion Data

### Context
- **Treatment (T)**: Peer-to-peer contact
- **Outcome (Y)**: Registration for reunion
- **Covariates (X)**: Alumni characteristics (years out, Greek affiliation, donor status, etc.)

### Assumptions to Verify

#### 1. Unconfoundedness
**Question**: Are all factors that influence BOTH "who gets contacted" AND "who registers" captured in features?

**Included** ✓:
- Alumni status
- Years out
- Donor status
- Greek affiliation
- Various constituencies

**Potential unmeasured confounders** ⚠️:
- Prior reunion attendance history
- Engagement level with university
- Geographic proximity to campus
- Personal relationship strength with volunteer callers
- Life circumstances (health, family, career status)

#### 2. Overlap
**Question**: Do you have comparable people in both contacted/not-contacted groups?

**Potential violations** ⚠️:
- Major donors might **always** be contacted (no overlap at high donor levels)
- Recent graduates might **never** be contacted (no overlap for recent years)
- Board members might be **always** contacted

**Check**: Create propensity score distribution plots to verify overlap

#### 3. SUTVA
**Question**: Does peer contact of one person affect another's registration?

**Potential violations** ⚠️:
- Alumni couples: If one is contacted, might influence the other
- Friend groups: Peer effects in decision-making
- Social networks: Word-of-mouth spreading beyond direct contact

#### 4. Consistency
**Question**: Is "peer contact" well-defined and consistent?

**Potential issues** ⚠️:
- Quality of contact varies (voicemail vs. conversation)
- Different volunteers have different effectiveness
- Timing of contact matters
- Multiple contacts vs. single contact

## Alternative Meta-Learners

### S-Learner (Single Model)
- **Method**: Single model with treatment as a feature
- **Pros**: More data efficient, simpler
- **Cons**: Assumes same features matter for treated/control

### X-Learner (Cross-Learner)
- **Method**: Uses propensity scores; estimates CATE more efficiently
- **Pros**: Better with imbalanced groups, more efficient
- **Cons**: More complex, requires propensity score estimation

### R-Learner (Robinson's Learner)
- **Method**: Directly models treatment effect after residualizing
- **Pros**: More robust to model misspecification
- **Cons**: Computationally intensive, requires careful implementation

### DR-Learner (Doubly Robust)
- **Method**: Combines outcome and propensity modeling
- **Pros**: Consistent if either model is correct
- **Cons**: Most complex, requires both models

## Practical Recommendations

### For Your Reunion Data:

1. **Check propensity scores**: 
   ```python
   from sklearn.linear_model import LogisticRegression
   propensity_model = LogisticRegression()
   propensity_model.fit(X_train[Predictors_no_treatment], treatment)
   propensity_scores = propensity_model.predict_proba(X_train[Predictors_no_treatment])[:,1]
   # Plot distribution by treatment group
   ```

2. **Validate overlap**:
   - Check for regions where propensity is near 0 or 1
   - Trim observations outside common support

3. **Sensitivity analysis**:
   - Try different base learners (XGBoost, Random Forest, etc.)
   - Compare T-Learner with X-Learner or S-Learner
   - Bootstrap confidence intervals for uplift estimates

4. **Domain validation**:
   - Do high uplift individuals make sense?
   - Share findings with reunion organizers for validation
   - Consider A/B testing recommendations

## Key Takeaway

**Unconfoundedness is the hardest assumption to verify** - you need domain knowledge to ensure all confounders are captured. The T-Learner will give you estimates, but whether they represent **causal effects** depends critically on whether this assumption holds.

If you suspect unmeasured confounding, consider:
- Collecting more data on potential confounders
- Using instrumental variables (if available)
- Conducting sensitivity analysis
- Being more cautious in causal interpretation

## References

- Künzel, S. R., Sekhon, J. S., Bickel, P. J., & Yu, B. (2019). "Metalearners for estimating heterogeneous treatment effects using machine learning." *PNAS*, 116(10), 4156-4165.
- Athey, S., & Imbens, G. W. (2016). "Recursive partitioning for heterogeneous causal effects." *PNAS*, 113(27), 7353-7360.
- Pearl, J. (2009). *Causality: Models, Reasoning and Inference* (2nd ed.). Cambridge University Press.
