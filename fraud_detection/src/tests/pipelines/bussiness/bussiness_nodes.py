"""
This is a boilerplate pipeline 'bussiness_metrics'
generated using Kedro 0.19.10
"""
from typing import Dict
from sklearn.metrics import confusion_matrix , roc_curve , auc , ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional
import pandas as pd
from typing import Optional
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
import mlflow


def generate_feature_importance(
    final_model_results: dict,
    X_train: pd.DataFrame
) -> Optional[pd.DataFrame]:

    model = final_model_results["model"]

    if not hasattr(model, "feature_importances_"):
        return None

    feature_importance = (
        pd.DataFrame({
            "feature": X_train.columns,
            "importance": model.feature_importances_
        })
        .sort_values("importance", ascending=False)
    )

    return feature_importance


def plot_feature_importance(
    feature_importance: Optional[pd.DataFrame],
    top_n: int = 10
) -> plt.Figure:
    """
    Generate a feature importance bar plot.

    Args:
        feature_importance: DataFrame with columns ['feature', 'importance']
        top_n: Number of top features to plot

    Returns:
        matplotlib Figure
    """

    if feature_importance is None or feature_importance.empty:
        raise ValueError("Feature importance data is empty or None.")

    top_features = feature_importance.head(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        top_features["feature"][::-1],
        top_features["importance"][::-1]
    )

    ax.set_title(f"Top {top_n} Feature Importances")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")

    plt.tight_layout()
    return fig

def generate_roc_auc_plot(
    final_model_results: dict
) -> plt.Figure:

    y_true = final_model_results["y_true"]
    y_pred_proba = final_model_results["predictions_proba"]

    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")

    return fig




def segment_portfolio(
    y_true,
    y_pred,
    loan_amounts,
    parameters: Dict = None
) -> Dict[str, Dict[str, float]]:
    
    # 1. Ensure everything is 1D (Flatten DataFrames/Series to Numpy Arrays)
    # This prevents the "Unable to coerce to Series" and "Truth value ambiguous" errors
    y_true_arr = np.array(y_true).flatten()
    y_pred_arr = np.array(y_pred).flatten()
    loan_amt_arr = np.array(loan_amounts).flatten()

    # 2. Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred_arr).ravel()

    # --- START: NEW CONFUSION MATRIX PLOT LOGIC ---
    labels = ["Rejected", "Approved"] if not parameters else parameters.get("labels", ["Rejected", "Approved"])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    # Re-calculate matrix for display to ensure proper format for the plotter
    cm = confusion_matrix(y_true_arr, y_pred_arr)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    
    disp.plot(ax=ax, cmap=plt.cm.Blues, values_format='d')
    plt.title("Loan Prediction Performance")
    
    # Save the plot locally and log to MLflow
    plot_filename = "confusion_matrix.png"
    plt.savefig(plot_filename)
    mlflow.log_artifact(plot_filename)
    plt.close(fig)

    # 3. Use boolean indexing on the flattened arrays
    segments = {
        "approved_good": {
            "count": int(tp),
            "amount": float(np.sum(loan_amt_arr[(y_pred_arr == 1) & (y_true_arr == 1)]))
        },
        "approved_bad": {
            "count": int(fp),
            "amount": float(np.sum(loan_amt_arr[(y_pred_arr == 1) & (y_true_arr == 0)]))
        },
        "rejected_good": {
            "count": int(fn),
            "amount": float(np.sum(loan_amt_arr[(y_pred_arr == 0) & (y_true_arr == 1)]))
        },
        "rejected_bad": {
            "count": int(tn),
            "amount": float(np.sum(loan_amt_arr[(y_pred_arr == 0) & (y_true_arr == 0)]))
        }
    }

    return segments



def loan_economics_parameters() -> Dict[str, float]:
    """
    Centralized loan paramsomics assumptions.
    Can be overridden via Kedro config.
    """

    return {
        "loan_term_years": 3.0,
        "interest_rate": 0.085,
        "operating_cost_rate": 0.025,
        "cost_of_capital": 0.04,
        "default_loss_rate": 0.65,
        "avg_time_to_default": 0.5
    }



def calculate_model_portfolio_financials(
    segments: Dict[str, Dict[str, float]],
    params: Dict[str, float]
) -> Dict[str, float]:
    """
    Compute financial performance of the model-approved portfolio.
    """

    good_amt = segments["approved_good"]["amount"]
    bad_amt = segments["approved_bad"]["amount"]

    # Revenue
    interest_good = good_amt * params["interest_rate"] * params["loan_term_years"]
    interest_bad = bad_amt * params["interest_rate"] * params["avg_time_to_default"]

    # Costs
    costs_good = good_amt * (
        params["operating_cost_rate"] + params["cost_of_capital"]
    ) * params["loan_term_years"]

    costs_bad = bad_amt * (
        params["operating_cost_rate"] + params["cost_of_capital"]
    ) * params["avg_time_to_default"]

    # Losses
    credit_losses = bad_amt * params["default_loss_rate"]

    net_profit = (
        interest_good
        + interest_bad
        - costs_good
        - costs_bad
        - credit_losses
    )

    total_approved = good_amt + bad_amt
    roi_annualized = (
        (net_profit / total_approved) / params["loan_term_years"] * 100
        if total_approved > 0
        else 0.0
    )

    return {
        "models_approved_good_amount": good_amt,
        "models_approved_bad_amount":bad_amt,
        "models_total_approved_amount": total_approved,
        "models_interest_over_good_amount": interest_good,
        "models_interest_over_bad_amount": interest_bad,
        "models_credit_losses": credit_losses,
        "models_net_profit": net_profit,
        "models_roi_annualized_pct": roi_annualized,
        
    }




def calculate_baseline_approve_all(
    y_true,
    loan_amounts,
    params: Dict[str, float]
) -> Dict[str, float]:
    """
    Baseline: approve all loans (no model).
    """
    # Ensure inputs are flat numpy arrays to avoid alignment/Series issues
    y_true_arr = np.array(y_true).flatten()
    loan_amt_arr = np.array(loan_amounts).flatten()

    good_amt = np.sum(loan_amt_arr[y_true_arr == 1])
    bad_amt = np.sum(loan_amt_arr[y_true_arr == 0])

    interest_good = good_amt * params["interest_rate"] * params["loan_term_years"]
    interest_bad = bad_amt * params["interest_rate"] * params["avg_time_to_default"]

    costs_good = good_amt * (
        params["operating_cost_rate"] + params["cost_of_capital"]
    ) * params["loan_term_years"]

    costs_bad = bad_amt * (
        params["operating_cost_rate"] + params["cost_of_capital"]
    ) * params["avg_time_to_default"]

    credit_losses = bad_amt * params["default_loss_rate"]

    net_profit = (
        interest_good
        + interest_bad
        - costs_good
        - costs_bad
        - credit_losses
    )

    total_amt = good_amt + bad_amt
    
    # Calculation for ROI
    if total_amt > 0:
        roi_annualized = (net_profit / total_amt) / params["loan_term_years"] * 100
    else:
        roi_annualized = 0.0

    # IMPORTANT: Convert to float() so Kedro can save to JSON
    return {
        "baseline_good_amount (amount for y_ture's approved)": float(good_amt),
        "baseline_interest_over_good_amount (int amount for y_ture's approved)": float(interest_good),
        "baseline_bad_amount (amount for y_ture's rejected)": float(good_amt),
        "baseline_interest_over_bad_amount (int amount for y_ture's rejected)": float(interest_bad),
        "baselien_total_amount" : float(total_amt),
        "baselien_credit_losses" : float(credit_losses),
        "baseline_net_profit": float(net_profit),
        "baseline_roi_annualized_pct": float(roi_annualized)
    }


def calculate_risk_metrics(
    segments: Dict[str, Dict[str, float]]
) -> Dict[str, float]:
    """
    Compute risk-related metrics.
    """

    approved_good = segments["approved_good"]["count"]
    approved_bad = segments["approved_bad"]["count"]

    total_approved = approved_good + approved_bad

    default_rate = (
        approved_bad / total_approved
        if total_approved > 0
        else 0.0
    )

    return {
        "approved_good" :approved_good,
        "approved_bad" :approved_bad,
        "total_approved" : total_approved,
        "approved_default_rate": default_rate
    }


def plot_business_summary(
    model_financials,
    baseline_financials
):
    """
    Visualization-only node.
    """

    labels = ["Baseline", "Model"]
    
    # Use .item() if it's a Series, or float() to ensure it's a scalar
    def to_scalar(val):
        if hasattr(val, "item"):
            return val.item() # Converts 1-element Series/Array to scalar
        return float(val)

    profits = [
        to_scalar(baseline_financials["baseline_net_profit"]),
        to_scalar(model_financials["models_net_profit"])
    ]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, profits, color=['gray', 'blue']) # Added color for clarity
    plt.title("Net Profit Comparison")
    plt.ylabel("Net Profit ($)")
    plt.tight_layout()
    
    # If using Kedro-Viz or notebooks, plt.show() is fine. 
    # If you want to save it as a Kedro artifact, you'll need to return the figure.
    return plt.gcf()


def log_business_metrics(
    model_financials,
    baseline_financials,
    risk_metrics
):
    """
    Centralized MLflow logging node.
    """

    mlflow.log_metric("model_net_profit", model_financials["models_net_profit"])
    mlflow.log_metric("model_roi_annualized_pct", model_financials["models_roi_annualized_pct"])

    mlflow.log_metric("baseline_net_profit", baseline_financials["baseline_net_profit"])
    mlflow.log_metric(
        "baseline_roi_annualized_pct",
        baseline_financials["baseline_roi_annualized_pct"]
    )

    mlflow.log_metric(
        "approved_default_rate",
        risk_metrics["approved_default_rate"]
    )



def create_business_dashboard(
    model_financials: Dict, 
    baseline_financials: Dict, 
    risk_metrics: Dict
):
    """
    Generates a 4-panel dashboard to explain the business value of the model.
    """
    # Set the style for a professional look
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Loan Model\'s Business Impact Dashboard', fontsize=20, fontweight='bold')

    # 1. Profit Comparison (Top Left)
    labels = ['Baseline (Approve All)', 'ML Model']
    profits = [baseline_financials["baseline_net_profit"], model_financials["models_net_profit"]]
    sns.barplot(x=labels, y=profits, ax=axes[0, 0], palette=['#A9A9A9', '#2E8B57'])
    axes[0, 0].set_title('Net Profit Comparison ($)', fontsize=14)
    axes[0, 0].set_ylabel('Profit')

    # 2. Credit Losses (Top Right)
    losses = [baseline_financials["baselien_credit_losses"], model_financials["models_credit_losses"]]
    sns.barplot(x=labels, y=losses, ax=axes[0, 1], palette=['#FF6347', '#CD5C5C'])
    axes[0, 1].set_title('Credit Losses (Lower is Better)', fontsize=14)
    axes[0, 1].set_ylabel('Loss Amount ($)')

    # 3. ROI Comparison (Bottom Left)
    roi = [baseline_financials["baseline_roi_annualized_pct"], model_financials["models_roi_annualized_pct"]]
    sns.barplot(x=labels, y=roi, ax=axes[1, 0], palette='viridis')
    axes[1, 0].set_title('Annualized ROI (%)', fontsize=14)
    axes[1, 0].set_ylabel('Percentage')

    # 4. Portfolio Composition (Bottom Right)
    # Showing Approved vs Rejected in the Model
    comp_labels = ['Approved Good', 'Approved Bad']
    comp_values = [risk_metrics["approved_good"], risk_metrics["approved_bad"]]
    axes[1, 1].pie(comp_values, labels=comp_labels, autopct='%1.1f%%', colors=['#4CAF50', '#F44336'], startangle=140)
    axes[1, 1].set_title('Model-Approved Portfolio Risk', fontsize=14)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save and Log to MLflow
    plot_path = "business_dashboard.png"
    plt.savefig(plot_path)
    #mlflow.log_artifact(plot_path)
    
    return fig










































































































#def calculate_business_impact(y_test, y_pred,  loan_amt_test):
#    """
#    Enhanced business metrics with realistic ROI calculations
#    
#    Key improvements:
#    1. Time-value adjusted ROI
#    2. Separate performing vs defaulted loan paramsomics
#    3. Multiple ROI scenarios (best case, expected, worst case)
#    4. Comparison with baseline (approve all) strategy
#    """
#    
#    # ========== BUSINESS PARAMETERS ==========
#    # Revenue parameters
#    AVG_LOAN_TERM_YEARS = 3  # Average loan duration
#    ANNUAL_INTEREST_RATE = 0.085  # 8.5% annual interest
#    
#    # Cost parameters
#    ANNUAL_OPERATING_COST_RATE = 0.025  # 2.5% annual operating costs
#    COST_OF_CAPITAL = 0.04  # 4% annual cost of capital
#    
#    # Default parameters
#    DEFAULT_LOSS_RATE = 0.65  # Lose 65% of principal on default
#    AVG_TIME_TO_DEFAULT_YEARS = 0.5  # Defaults happen within 6 months on average
#    RECOVERY_RATE = 1 - DEFAULT_LOSS_RATE  # 35% recovery
#    
#    # ========== DATA PREPARATION ==========
#    if isinstance(y_test, pd.Series):
#        y_test = y_test.reset_index(drop=True)
#    if isinstance(loan_amt_test, pd.Series):
#        loan_amt_test = loan_amt_test.reset_index(drop=True)
#    
#    y_true = np.array(y_test).flatten()
#    y_pred_arr = np.array(y_pred).flatten()
#    loan_amounts = np.array(loan_amt_test).flatten()
#    
#    print(f"\n{'='*80}")
#    print(f"ENHANCED BUSINESS IMPACT ANALYSIS")
#    print(f"{'='*80}")
#    
#    # ========== CONFUSION MATRIX ==========
#    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_arr).ravel()
#    
#    # Create masks
#    approved_mask = (y_pred_arr == 1)
#    rejected_mask = (y_pred_arr == 0)
#    actually_good_mask = (y_true == 1)
#    actually_bad_mask = (y_true == 0)
#    
#    # Loan segments
#    approved_good = approved_mask & actually_good_mask  # TP: Will perform
#    approved_bad = approved_mask & actually_bad_mask    # FP: Will default
#    rejected_good = rejected_mask & actually_good_mask  # FN: Missed opportunity
#    rejected_bad = rejected_mask & actually_bad_mask    # TN: Avoided loss
#    
#    # Counts and amounts
#    n_approved_good = np.sum(approved_good)
#    n_approved_bad = np.sum(approved_bad)
#    
#    amt_approved_good = np.sum(loan_amounts[approved_good])
#    amt_approved_bad = np.sum(loan_amounts[approved_bad])
#    total_approved_amount = amt_approved_good + amt_approved_bad
#    
#    # ========== SCENARIO 1: PERFORMING LOANS (Good Loans We Approved) ==========
#    print(f"\n📊 PERFORMING LOAN PORTFOLIO (Good Loans - Will Pay Back)")
#    print(f"{'─'*80}")
#    
#    if n_approved_good > 0:
#        # Revenue from performing loans over full term
#        performing_total_interest = amt_approved_good * ANNUAL_INTEREST_RATE * AVG_LOAN_TERM_YEARS
#        performing_operating_costs = amt_approved_good * ANNUAL_OPERATING_COST_RATE * AVG_LOAN_TERM_YEARS
#        performing_capital_costs = amt_approved_good * COST_OF_CAPITAL * AVG_LOAN_TERM_YEARS
#        performing_net_profit = performing_total_interest - performing_operating_costs - performing_capital_costs
#        performing_roi = (performing_net_profit / amt_approved_good) * 100
#        
#        print(f"  Number of Loans:        {n_approved_good:>8,}")
#        print(f"  Total Amount:           ${amt_approved_good:>15,.2f}")
#        print(f"  Interest Revenue:       ${performing_total_interest:>15,.2f}")
#        print(f"  Operating Costs:        ${performing_operating_costs:>15,.2f}")
#        print(f"  Capital Costs:          ${performing_capital_costs:>15,.2f}")
#        print(f"  Net Profit:             ${performing_net_profit:>15,.2f}")
#        print(f"  ROI (3-year):           {performing_roi:>15.2f}%")
#        print(f"  Annualized ROI:         {performing_roi/AVG_LOAN_TERM_YEARS:>15.2f}%")
#    else:
#        performing_net_profit = 0
#        performing_roi = 0
#    
#    # ========== SCENARIO 2: DEFAULTED LOANS (Bad Loans We Approved) ==========
#    print(f"\n❌ DEFAULTED LOAN PORTFOLIO (Bad Loans - Will Default)")
#    print(f"{'─'*80}")
#    
#    if n_approved_bad > 0:
#        # Revenue before default (partial interest collected)
#        default_partial_interest = amt_approved_bad * ANNUAL_INTEREST_RATE * AVG_TIME_TO_DEFAULT_YEARS
#        default_operating_costs = amt_approved_bad * ANNUAL_OPERATING_COST_RATE * AVG_TIME_TO_DEFAULT_YEARS
#        default_capital_costs = amt_approved_bad * COST_OF_CAPITAL * AVG_TIME_TO_DEFAULT_YEARS
#        
#        # Principal loss
#        principal_loss = amt_approved_bad * DEFAULT_LOSS_RATE
#        recovery_amount = amt_approved_bad * RECOVERY_RATE
#        
#        default_net_loss = principal_loss + default_operating_costs + default_capital_costs - default_partial_interest
#        default_roi = -(default_net_loss / amt_approved_bad) * 100
#        
#        print(f"  Number of Loans:        {n_approved_bad:>8,}")
#        print(f"  Total Amount:           ${amt_approved_bad:>15,.2f}")
#        print(f"  Partial Interest:       ${default_partial_interest:>15,.2f}")
#        print(f"  Principal Loss:         ${principal_loss:>15,.2f}")
#        print(f"  Recovery:               ${recovery_amount:>15,.2f}")
#        print(f"  Operating Costs:        ${default_operating_costs:>15,.2f}")
#        print(f"  Capital Costs:          ${default_capital_costs:>15,.2f}")
#        print(f"  Net Loss:               ${default_net_loss:>15,.2f}")
#        print(f"  ROI:                    {default_roi:>15.2f}%")
#    else:
#        default_net_loss = 0
#        default_roi = 0
#        default_partial_interest = 0
#    
#    # ========== COMBINED PORTFOLIO paramsOMICS ==========
#    print(f"\n{'='*80}")
#    print(f"💼 COMBINED PORTFOLIO (3-YEAR PROJECTION)")
#    print(f"{'='*80}")
#    
#    # Total revenue (full term for good loans, partial for defaults)
#    if n_approved_good > 0:
#        total_revenue = performing_total_interest + (default_partial_interest if n_approved_bad > 0 else 0)
#    else:
#        total_revenue = 0
#    
#    # Total costs
#    if n_approved_good > 0:
#        total_operating = performing_operating_costs + (default_operating_costs if n_approved_bad > 0 else 0)
#        total_capital = performing_capital_costs + (default_capital_costs if n_approved_bad > 0 else 0)
#    else:
#        total_operating = 0
#        total_capital = 0
#    
#    total_default_losses = principal_loss if n_approved_bad > 0 else 0
#    
#    # Net profit/loss
#    total_net_profit = total_revenue - total_operating - total_capital - total_default_losses
#    blended_roi = (total_net_profit / total_approved_amount * 100) if total_approved_amount > 0 else 0
#    annualized_roi = blended_roi / AVG_LOAN_TERM_YEARS
#    
#    print(f"\n📈 REVENUE")
#    print(f"  Total Interest Income:      ${total_revenue:>15,.2f}")
#    
#    print(f"\n📉 COSTS")
#    print(f"  Operating Costs:            ${total_operating:>15,.2f}")
#    print(f"  Capital Costs:              ${total_capital:>15,.2f}")
#    print(f"  Default Losses:             ${total_default_losses:>15,.2f}")
#    print(f"  {'─'*50}")
#    print(f"  Total Costs:                ${(total_operating + total_capital + total_default_losses):>15,.2f}")
#    
#    print(f"\n{'═'*80}")
#    print(f"💰 NET PROFIT (3-YEAR):       ${total_net_profit:>15,.2f}")
#    print(f"📊 BLENDED ROI (3-YEAR):      {blended_roi:>15.2f}%")
#    print(f"📊 ANNUALIZED ROI:            {annualized_roi:>15.2f}%")
#    print(f"{'═'*80}")
#    
#    # ========== BASELINE COMPARISON: WHAT IF WE APPROVED ALL LOANS? ==========
#    print(f"\n🔍 BASELINE COMPARISON: Approve All Strategy")
#    print(f"{'─'*80}")
#    
#    total_loans = len(y_true)
#    n_actually_good = np.sum(actually_good_mask)
#    n_actually_bad = np.sum(actually_bad_mask)
#    amt_all_good = np.sum(loan_amounts[actually_good_mask])
#    amt_all_bad = np.sum(loan_amounts[actually_bad_mask])
#    total_all_amounts = np.sum(loan_amounts)
#    
#    # Revenue from all good loans
#    baseline_good_revenue = amt_all_good * ANNUAL_INTEREST_RATE * AVG_LOAN_TERM_YEARS
#    baseline_good_costs = amt_all_good * (ANNUAL_OPERATING_COST_RATE + COST_OF_CAPITAL) * AVG_LOAN_TERM_YEARS
#    
#    # Losses from all bad loans
#    baseline_bad_revenue = amt_all_bad * ANNUAL_INTEREST_RATE * AVG_TIME_TO_DEFAULT_YEARS
#    baseline_bad_costs = amt_all_bad * (ANNUAL_OPERATING_COST_RATE + COST_OF_CAPITAL) * AVG_TIME_TO_DEFAULT_YEARS
#    baseline_bad_losses = amt_all_bad * DEFAULT_LOSS_RATE
#    
#    baseline_net_profit = (baseline_good_revenue - baseline_good_costs + 
#                           baseline_bad_revenue - baseline_bad_costs - baseline_bad_losses)
#    baseline_roi = (baseline_net_profit / total_all_amounts) * 100
#    baseline_annualized_roi = baseline_roi / AVG_LOAN_TERM_YEARS
#    
#    print(f"  Total Loans:                {total_loans:>8,}")
#    print(f"  Good Loans:                 {n_actually_good:>8,} (${amt_all_good:,.2f})")
#    print(f"  Bad Loans:                  {n_actually_bad:>8,} (${amt_all_bad:,.2f})")
#    print(f"  Net Profit:                 ${baseline_net_profit:>15,.2f}")
#    print(f"  Blended ROI (3-year):       {baseline_roi:>15.2f}%")
#    print(f"  Annualized ROI:             {baseline_annualized_roi:>15.2f}%")
#    
#    # ========== VALUE CREATED BY MODEL ==========
#    value_vs_baseline = total_net_profit - baseline_net_profit
#    roi_improvement = blended_roi - baseline_roi
#    
#    print(f"\n{'='*80}")
#    print(f"🎯 VALUE CREATED BY MODEL")
#    print(f"{'='*80}")
#    print(f"  Profit Improvement:         ${value_vs_baseline:>15,.2f}")
#    print(f"  ROI Improvement:            {roi_improvement:>15.2f} percentage points")
#    
#    # Calculate opportunity cost
#    amt_rejected_good = np.sum(loan_amounts[rejected_good])
#    n_rejected_good = np.sum(rejected_good)
#    if n_rejected_good > 0:
#        opportunity_profit = (amt_rejected_good * ANNUAL_INTEREST_RATE * AVG_LOAN_TERM_YEARS - 
#                             amt_rejected_good * (ANNUAL_OPERATING_COST_RATE + COST_OF_CAPITAL) * AVG_LOAN_TERM_YEARS)
#        print(f"  Opportunity Cost:           ${opportunity_profit:>15,.2f} ({n_rejected_good:,} good loans rejected)")
#    else:
#        opportunity_profit = 0
#    
#    # Calculate losses avoided
#    amt_rejected_bad = np.sum(loan_amounts[rejected_bad])
#    n_rejected_bad = np.sum(rejected_bad)
#    if n_rejected_bad > 0:
#        losses_avoided = amt_rejected_bad * DEFAULT_LOSS_RATE
#        print(f"  Losses Avoided:             ${losses_avoided:>15,.2f} ({n_rejected_bad:,} bad loans rejected)")
#    else:
#        losses_avoided = 0
#    
#    net_value_created = total_net_profit + losses_avoided - opportunity_profit
#    print(f"  {'─'*50}")
#    print(f"  NET VALUE CREATED:          ${net_value_created:>15,.2f}")
#    print(f"{'='*80}")
#    
#    # ========== RISK METRICS ==========
#    default_rate_portfolio = (n_approved_bad / (n_approved_good + n_approved_bad) * 100) if (n_approved_good + n_approved_bad) > 0 else 0
#    default_rate_baseline = (n_actually_bad / total_loans) * 100
#    
#    print(f"\n⚠️  RISK ANALYSIS")
#    print(f"{'─'*80}")
#    print(f"  Model Default Rate:         {default_rate_portfolio:>6.2f}%")
#    print(f"  Baseline Default Rate:      {default_rate_baseline:>6.2f}%")
#    print(f"  Risk Reduction:             {default_rate_baseline - default_rate_portfolio:>6.2f} percentage points")
#    
#    # ========== COMPILE ALL METRICS ==========
#    metrics = {
#        # Volume
#        'total_loans': int(total_loans),
#        'loans_approved': int(n_approved_good + n_approved_bad),
#        'loans_rejected': int(n_rejected_good + n_rejected_bad),
#        'approval_rate_pct': float((n_approved_good + n_approved_bad) / total_loans * 100),
#        
#        # Approved breakdown
#        'approved_good': int(n_approved_good),
#        'approved_bad': int(n_approved_bad),
#        'rejected_good': int(n_rejected_good),
#        'rejected_bad': int(n_rejected_bad),
#        
#        # Financial - Model Portfolio (3-year)
#        'model_total_revenue': float(total_revenue),
#        'model_total_costs': float(total_operating + total_capital),
#        'model_default_losses': float(total_default_losses),
#        'model_net_profit': float(total_net_profit),
#        'model_roi_3year_pct': float(blended_roi),
#        'model_roi_annualized_pct': float(annualized_roi),
#        
#        # Financial - Baseline (3-year)
#        'baseline_net_profit': float(baseline_net_profit),
#        'baseline_roi_3year_pct': float(baseline_roi),
#        'baseline_roi_annualized_pct': float(baseline_annualized_roi),
#        
#        # Value creation
#        'profit_improvement': float(value_vs_baseline),
#        'roi_improvement_pct': float(roi_improvement),
#        'opportunity_cost': float(opportunity_profit),
#        'losses_avoided': float(losses_avoided),
#        'net_value_created': float(net_value_created),
#        
#        # Risk
#        'model_default_rate_pct': float(default_rate_portfolio),
#        'baseline_default_rate_pct': float(default_rate_baseline),
#        'risk_reduction_pct': float(default_rate_baseline - default_rate_portfolio),
#        
#        # Performance
#        'precision': float(tp / (tp + fp) if (tp + fp) > 0 else 0),
#        'recall': float(tp / (tp + fn) if (tp + fn) > 0 else 0),
#        'specificity': float(tn / (tn + fp) if (tn + fp) > 0 else 0),
#        'accuracy': float((tp + tn) / total_loans),
#    }
#    
#    # Log to MLflow
#    #try:
#    #    import mlflow
#    #    if mlflow.active_run():
#    #        for key, value in metrics.items():
#    #            mlflow.log_metric(f"enhanced_{key}", float(value))
#    #        print("\n✅ Enhanced metrics logged to MLflow")
#    #except Exception as e:
#    #    print(f"\nNote: MLflow logging skipped - {e}")
#    
#    return pd.DataFrame([metrics])
#
#
#
#def visualize_business_metrics(enhanced_business_report):
#    """
#    Create comprehensive business metrics visualization dashboard
#    Optimized for the enhanced business report with time-adjusted ROI
#    
#    Args:
#        enhanced_business_report: DataFrame with enhanced business metrics
#    
#    Returns:
#        matplotlib.figure.Figure: Comprehensive dashboard figure
#    """
#    # Extract metrics
#    m = enhanced_business_report.iloc[0].to_dict()
#    
#    # Set style
#    sns.set_style("whitegrid")
#    plt.rcParams['font.family'] = 'sans-serif'
#    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
#    
#    # Create figure with custom grid
#    fig = plt.figure(figsize=(24, 16))
#    gs = GridSpec(4, 4, figure=fig, hspace=0.35, wspace=0.35, 
#                  left=0.05, right=0.95, top=0.92, bottom=0.05)
#    
#    # Color palette
#    colors = {
#        'profit': '#27ae60',
#        'loss': '#e74c3c',
#        'neutral': '#3498db',
#        'warning': '#f39c12',
#        'sparamsdary': '#95a5a6',
#        'dark': '#2c3e50',
#        'light_green': '#a8e6cf',
#        'light_red': '#ffaaa5',
#        'light_blue': '#a8d8ea'
#    }
#    
#    # ========== 1. EXECUTIVE SUMMARY PANEL (Top Span) ==========
#    ax_summary = fig.add_subplot(gs[0, :])
#    ax_summary.axis('off')
#    
#    # Main title
#    fig.text(0.5, 0.97, '🏦 LOAN PREDICTION MODEL - BUSINESS IMPACT DASHBOARD', 
#             ha='center', fontsize=22, fontweight='bold', 
#             bbox=dict(boxstyle='round,pad=0.5', facecolor=colors['neutral'], 
#                      alpha=0.2, edgecolor=colors['dark'], linewidth=2))
#    
#    # Executive summary boxes
#    summary_metrics = [
#        {
#            'title': 'Model Performance',
#            'value': f"{m['model_roi_annualized_pct']:.2f}%",
#            'subtitle': 'Annualized ROI',
#            'icon': '📈',
#            'color': colors['profit'] if m['model_roi_annualized_pct'] > 0 else colors['loss']
#        },
#        {
#            'title': 'Value Creation',
#            'value': f"${m['net_value_created']/1e6:.2f}M",
#            'subtitle': 'Net Value Added',
#            'icon': '💰',
#            'color': colors['profit']
#        },
#        {
#            'title': 'Risk Reduction',
#            'value': f"{m['risk_reduction_pct']:.1f}pp",
#            'subtitle': 'Default Rate Reduction',
#            'icon': '🛡️',
#            'color': colors['profit']
#        },
#        {
#            'title': 'Portfolio Quality',
#            'value': f"{m['precision']*100:.1f}%",
#            'subtitle': 'Precision Score',
#            'icon': '🎯',
#            'color': colors['neutral']
#        },
#        {
#            'title': 'Coverage',
#            'value': f"{m['recall']*100:.1f}%",
#            'subtitle': 'Recall Score',
#            'icon': '📊',
#            'color': colors['neutral']
#        }
#    ]
#    
#    box_width = 0.18
#    x_start = 0.05
#    for i, metric in enumerate(summary_metrics):
#        x = x_start + i * 0.19
#        
#        # Background box
#        fancy_box = FancyBboxPatch((x, 0.15), box_width, 0.7,
#                                   boxstyle="round,pad=0.02",
#                                   facecolor=metric['color'], alpha=0.15,
#                                   edgecolor=metric['color'], linewidth=2.5,
#                                   transform=ax_summary.transAxes)
#        ax_summary.add_patch(fancy_box)
#        
#        # Icon
#        ax_summary.text(x + box_width/2, 0.75, metric['icon'],
#                       ha='center', va='center', fontsize=28,
#                       transform=ax_summary.transAxes)
#        
#        # Value
#        ax_summary.text(x + box_width/2, 0.55, metric['value'],
#                       ha='center', va='center', fontsize=18, fontweight='bold',
#                       color=metric['color'], transform=ax_summary.transAxes)
#        
#        # Subtitle
#        ax_summary.text(x + box_width/2, 0.38, metric['subtitle'],
#                       ha='center', va='center', fontsize=10,
#                       style='italic', transform=ax_summary.transAxes)
#        
#        # Title
#        ax_summary.text(x + box_width/2, 0.22, metric['title'],
#                       ha='center', va='center', fontsize=11, fontweight='bold',
#                       transform=ax_summary.transAxes)
#    
#    # ========== 2. ROI COMPARISON (Top Left) ==========
#    ax_roi = fig.add_subplot(gs[1, 0])
#    
#    roi_categories = ['Model\nPortfolio', 'Approve All\nBaseline', 'Improvement']
#    roi_values = [
#        m['model_roi_annualized_pct'],
#        m['baseline_roi_annualized_pct'],
#        m['roi_improvement_pct'] / 3  # Annualized
#    ]
#    roi_colors = [
#        colors['profit'] if roi_values[0] > 0 else colors['loss'],
#        colors['loss'] if roi_values[1] < 0 else colors['sparamsdary'],
#        colors['profit']
#    ]
#    
#    bars = ax_roi.bar(roi_categories, roi_values, color=roi_colors, 
#                      alpha=0.7, edgecolor='black', linewidth=2, width=0.6)
#    
#    # Add horizontal line at 0
#    ax_roi.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
#    
#    # Add value labels
#    for bar, val in zip(bars, roi_values):
#        height = bar.get_height()
#        ax_roi.text(bar.get_x() + bar.get_width()/2., height,
#                   f'{val:.2f}%',
#                   ha='center', va='bottom' if height > 0 else 'top',
#                   fontsize=12, fontweight='bold',
#                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
#    
#    ax_roi.set_ylabel('Annualized ROI (%)', fontsize=13, fontweight='bold')
#    ax_roi.set_title('📊 ROI Comparison (Annualized)', fontsize=14, fontweight='bold', pad=15)
#    ax_roi.grid(axis='y', alpha=0.3, linestyle='--')
#    ax_roi.spines['top'].set_visible(False)
#    ax_roi.spines['right'].set_visible(False)
#    
#    # ========== 3. 3-YEAR PROFIT BREAKDOWN (Top Center-Left) ==========
#    ax_profit = fig.add_subplot(gs[1, 1])
#    
#    categories = ['Revenue', 'Operating\nCosts', 'Capital\nCosts', 'Default\nLosses', 'Net\nProfit']
#    values = [
#        m['model_total_revenue'],
#        -m['model_total_costs'],
#        0,  # Already included in total costs
#        -m['model_default_losses'],
#        m['model_net_profit']
#    ]
#    colors_list = [colors['profit'], colors['loss'], colors['loss'], 
#                   colors['loss'], colors['profit'] if m['model_net_profit'] > 0 else colors['loss']]
#    
#    # Filter out zero values
#    non_zero = [(c, v, col) for c, v, col in zip(categories, values, colors_list) if v != 0]
#    if non_zero:
#        cats, vals, cols = zip(*non_zero)
#        
#        bars = ax_profit.barh(cats, vals, color=cols, alpha=0.7, 
#                             edgecolor='black', linewidth=2)
#        
#        # Add value labels
#        for bar, val in zip(bars, vals):
#            width = bar.get_width()
#            ax_profit.text(width, bar.get_y() + bar.get_height()/2.,
#                          f'${abs(val)/1e6:.2f}M',
#                          ha='left' if width > 0 else 'right',
#                          va='center', fontsize=11, fontweight='bold',
#                          bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
#    
#    ax_profit.axvline(x=0, color='black', linestyle='-', linewidth=1.5)
#    ax_profit.set_xlabel('Amount ($ Millions)', fontsize=13, fontweight='bold')
#    ax_profit.set_title('💰 3-Year Financial Breakdown', fontsize=14, fontweight='bold', pad=15)
#    ax_profit.grid(axis='x', alpha=0.3, linestyle='--')
#    ax_profit.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
#    
#    # ========== 4. CONFUSION MATRIX HEATMAP (Top Center-Right) ==========
#    ax_conf = fig.add_subplot(gs[1, 2])
#    
#    confusion_data = np.array([
#        [m['rejected_bad'], m['rejected_good']],
#        [m['approved_bad'], m['approved_good']]
#    ])
#    
#    # Custom colormap
#    im = ax_conf.imshow(confusion_data, cmap='RdYlGn', aspect='auto', alpha=0.7)
#    
#    # Add text annotations
#    for i in range(2):
#        for j in range(2):
#            text = ax_conf.text(j, i, f"{confusion_data[i, j]:,}",
#                              ha="center", va="center", fontsize=16, 
#                              fontweight='bold', color='black')
#    
#    # Labels and formatting
#    ax_conf.set_xticks([0, 1])
#    ax_conf.set_yticks([0, 1])
#    ax_conf.set_xticklabels(['Bad Loans\n(Will Default)', 'Good Loans\n(Will Perform)'], 
#                           fontsize=11, fontweight='bold')
#    ax_conf.set_yticklabels(['Rejected', 'Approved'], fontsize=11, fontweight='bold')
#    ax_conf.set_xlabel('Actual Outcome', fontsize=12, fontweight='bold', labelpad=10)
#    ax_conf.set_ylabel('Model Decision', fontsize=12, fontweight='bold', labelpad=10)
#    ax_conf.set_title('🎯 Decision Matrix', fontsize=14, fontweight='bold', pad=15)
#    
#    # Add emoji annotations
#    emoji_positions = [
#        (0, 0, '✓', colors['profit']),
#        (1, 0, '⚠️', colors['warning']),
#        (0, 1, '✗', colors['loss']),
#        (1, 1, '✓', colors['profit'])
#    ]
#    for i, j, emoji, color in emoji_positions:
#        ax_conf.text(j, i-0.35, emoji, ha='center', va='center', fontsize=20)
#    
#    # Colorbar
#    cbar = plt.colorbar(im, ax=ax_conf, fraction=0.046, pad=0.04)
#    cbar.set_label('Number of Loans', fontsize=10, fontweight='bold')
#    
#    # ========== 5. VALUE CREATION WATERFALL (Top Right) ==========
#    ax_waterfall = fig.add_subplot(gs[1, 3])
#    
#    # Waterfall data
#    waterfall_data = [
#        ('Model\nProfit', m['model_net_profit'], colors['profit'] if m['model_net_profit'] > 0 else colors['loss']),
#        ('Losses\nAvoided', m['losses_avoided'], colors['profit']),
#        ('Opportunity\nCost', -m['opportunity_cost'], colors['warning']),
#    ]
#    
#    cumulative = 0
#    positions = []
#    heights = []
#    colors_wf = []
#    bottoms = []
#    
#    for label, value, color in waterfall_data:
#        positions.append(label)
#        heights.append(abs(value))
#        colors_wf.append(color)
#        bottoms.append(cumulative if value > 0 else cumulative + value)
#        cumulative += value
#    
#    # Add final bar
#    positions.append('Net\nValue')
#    heights.append(abs(cumulative))
#    colors_wf.append(colors['profit'])
#    bottoms.append(0 if cumulative > 0 else cumulative)
#    
#    bars = ax_waterfall.bar(range(len(positions)), heights, bottom=bottoms,
#                            color=colors_wf, alpha=0.7, edgecolor='black', linewidth=2)
#    
#    # Connect bars with lines
#    for i in range(len(positions)-1):
#        prev_top = bottoms[i] + heights[i]
#        next_bottom = bottoms[i+1]
#        ax_waterfall.plot([i+0.4, i+0.6], [prev_top, next_bottom], 
#                         'k--', linewidth=1.5, alpha=0.5)
#    
#    # Add value labels
#    for i, (bar, height, bottom) in enumerate(zip(bars, heights, bottoms)):
#        value = height if i < len(waterfall_data) else cumulative
#        ax_waterfall.text(bar.get_x() + bar.get_width()/2., 
#                         bottom + height,
#                         f'${abs(value)/1e6:.2f}M',
#                         ha='center', va='bottom', fontsize=11, fontweight='bold',
#                         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
#    
#    ax_waterfall.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
#    ax_waterfall.set_xticks(range(len(positions)))
#    ax_waterfall.set_xticklabels(positions, fontsize=11, fontweight='bold')
#    ax_waterfall.set_ylabel('Value ($)', fontsize=13, fontweight='bold')
#    ax_waterfall.set_title('🌊 Value Creation Waterfall', fontsize=14, fontweight='bold', pad=15)
#    ax_waterfall.grid(axis='y', alpha=0.3, linestyle='--')
#    ax_waterfall.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
#    
#    # ========== 6. PORTFOLIO COMPOSITION PIE CHARTS (Middle Left) ==========
#    ax_pie = fig.add_subplot(gs[2, 0])
#    
#    # Model portfolio
#    model_sizes = [m['approved_good'], m['approved_bad']]
#    model_labels = [f"Performing\n{m['approved_good']:,}\n({m['approved_good']/(m['approved_good']+m['approved_bad'])*100:.1f}%)",
#                   f"Defaulted\n{m['approved_bad']:,}\n({m['approved_bad']/(m['approved_good']+m['approved_bad'])*100:.1f}%)"]
#    
#    wedges, texts = ax_pie.pie(model_sizes, labels=model_labels, 
#                                colors=[colors['profit'], colors['loss']],
#                                startangle=90, counterclock=False,
#                                wedgeprops=dict(edgecolor='black', linewidth=2, alpha=0.7),
#                                textprops={'fontsize': 11, 'fontweight': 'bold'})
#    
#    ax_pie.set_title(f'📊 Model Portfolio Composition\n(Total: {m["loans_approved"]:,} loans approved)', 
#                    fontsize=13, fontweight='bold', pad=15)
#    
#    # ========== 7. MODEL PERFORMANCE RADAR (Middle Center-Left) ==========
#    ax_radar = fig.add_subplot(gs[2, 1], projection='polar')
#    
#    categories_radar = ['Precision', 'Recall', 'Specificity', 'Accuracy', 'F1-Score']
#    values_radar = [
#        m['precision'] * 100,
#        m['recall'] * 100,
#        m['specificity'] * 100,
#        m['accuracy'] * 100,
#        2 * (m['precision'] * m['recall']) / (m['precision'] + m['recall']) * 100 if (m['precision'] + m['recall']) > 0 else 0
#    ]
#    
#    # Number of variables
#    N = len(categories_radar)
#    angles = [n / float(N) * 2 * np.pi for n in range(N)]
#    values_radar += values_radar[:1]
#    angles += angles[:1]
#    
#    ax_radar.plot(angles, values_radar, 'o-', linewidth=2, color=colors['neutral'])
#    ax_radar.fill(angles, values_radar, alpha=0.25, color=colors['neutral'])
#    ax_radar.set_xticks(angles[:-1])
#    ax_radar.set_xticklabels(categories_radar, fontsize=10, fontweight='bold')
#    ax_radar.set_ylim(0, 100)
#    ax_radar.set_yticks([20, 40, 60, 80, 100])
#    ax_radar.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=8)
#    ax_radar.grid(True, linestyle='--', alpha=0.5)
#    ax_radar.set_title('🎯 Model Performance Metrics', fontsize=13, fontweight='bold', 
#                      pad=20, y=1.08)
#    
#    # ========== 8. RISK COMPARISON (Middle Center-Right) ==========
#    ax_risk = fig.add_subplot(gs[2, 2])
#    
#    risk_categories = ['Model\nPortfolio', 'Population\nBaseline']
#    risk_values = [m['model_default_rate_pct'], m['baseline_default_rate_pct']]
#    risk_colors = [colors['profit'], colors['loss']]
#    
#    bars = ax_risk.bar(risk_categories, risk_values, color=risk_colors, 
#                       alpha=0.7, edgecolor='black', linewidth=2, width=0.5)
#    
#    # Add value labels
#    for bar, val in zip(bars, risk_values):
#        height = bar.get_height()
#        ax_risk.text(bar.get_x() + bar.get_width()/2., height,
#                    f'{val:.2f}%',
#                    ha='center', va='bottom', fontsize=14, fontweight='bold',
#                    bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9))
#    
#    # Add reduction arrow
#    ax_risk.annotate('', xy=(1, risk_values[1]), xytext=(1, risk_values[0]),
#                    arrowprops=dict(arrowstyle='<->', color=colors['profit'], 
#                                  lw=3, ls='--'))
#    ax_risk.text(1.3, (risk_values[0] + risk_values[1])/2, 
#                f'{m["risk_reduction_pct"]:.1f}pp\nreduction',
#                fontsize=11, fontweight='bold', color=colors['profit'],
#                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
#                         edgecolor=colors['profit'], alpha=0.9, linewidth=2))
#    
#    ax_risk.set_ylabel('Default Rate (%)', fontsize=13, fontweight='bold')
#    ax_risk.set_title('⚠️ Default Rate Comparison', fontsize=14, fontweight='bold', pad=15)
#    ax_risk.grid(axis='y', alpha=0.3, linestyle='--')
#    ax_risk.set_ylim(0, max(risk_values) * 1.3)
#    
#    # ========== 9. LOAN VOLUME DISTRIBUTION (Middle Right) ==========
#    ax_volume = fig.add_subplot(gs[2, 3])
#    
#    categories_vol = ['Good\nLoans', 'Bad\nLoans']
#    approved_vals = [m['approved_good'], m['approved_bad']]
#    rejected_vals = [m['rejected_good'], m['rejected_bad']]
#    
#    x = np.arange(len(categories_vol))
#    width = 0.35
#    
#    bars1 = ax_volume.bar(x - width/2, approved_vals, width, label='Approved',
#                         color=colors['profit'], alpha=0.7, edgecolor='black', linewidth=2)
#    bars2 = ax_volume.bar(x + width/2, rejected_vals, width, label='Rejected',
#                         color=colors['loss'], alpha=0.7, edgecolor='black', linewidth=2)
#    
#    # Add value labels
#    for bars in [bars1, bars2]:
#        for bar in bars:
#            height = bar.get_height()
#            ax_volume.text(bar.get_x() + bar.get_width()/2., height,
#                          f'{int(height):,}',
#                          ha='center', va='bottom', fontsize=10, fontweight='bold')
#    
#    ax_volume.set_ylabel('Number of Loans', fontsize=13, fontweight='bold')
#    ax_volume.set_title('📦 Loan Volume Distribution', fontsize=14, fontweight='bold', pad=15)
#    ax_volume.set_xticks(x)
#    ax_volume.set_xticklabels(categories_vol, fontsize=11, fontweight='bold')
#    ax_volume.legend(fontsize=11, loc='upper right', framealpha=0.9)
#    ax_volume.grid(axis='y', alpha=0.3, linestyle='--')
#    
#    # ========== 10. KEY INSIGHTS BOX (Bottom Span) ==========
#    ax_insights = fig.add_subplot(gs[3, :])
#    ax_insights.axis('off')
#    
#    # Calculate key insights
#    approval_rate = m['approval_rate_pct']
#    roi_text = f"+{m['model_roi_annualized_pct']:.2f}%" if m['model_roi_annualized_pct'] > 0 else f"{m['model_roi_annualized_pct']:.2f}%"
#    precision_text = f"{m['precision']*100:.1f}%"
#    
#    # Create insight boxes
#    insights = [
#        {
#            'icon': '🎯',
#            'title': 'Model Strategy',
#            'text': f"Conservative approval strategy ({approval_rate:.1f}% approval rate) prioritizes risk mitigation over volume growth."
#        },
#        {
#            'icon': '💰',
#            'title': 'Financial Performance',
#            'text': f"Annualized ROI of {roi_text} demonstrates profitable operations with {precision_text} of approved loans performing well."
#        },
#        {
#            'icon': '🛡️',
#            'title': 'Risk Management',
#            'text': f"Model reduces default rate by {m['risk_reduction_pct']:.1f} percentage points, avoiding ${m['losses_avoided']/1e6:.1f}M in potential losses."
#        },
#        {
#            'icon': '📈',
#            'title': 'Growth Opportunity',
#            'text': f"${m['opportunity_cost']/1e6:.2f}M opportunity cost from {m['rejected_good']:,} rejected good loans suggests room for threshold optimization."
#        }
#    ]
#    
#    box_width = 0.23
#    x_start = 0.02
#    for i, insight in enumerate(insights):
#        x = x_start + i * 0.245
#        
#        # Background box
#        fancy_box = FancyBboxPatch((x, 0.1), box_width, 0.75,
#                                   boxstyle="round,pad=0.02",
#                                   facecolor=colors['neutral'], alpha=0.1,
#                                   edgecolor=colors['dark'], linewidth=2,
#                                   transform=ax_insights.transAxes)
#        ax_insights.add_patch(fancy_box)
#        
#        # Icon
#        ax_insights.text(x + 0.03, 0.75, insight['icon'],
#                        ha='left', va='top', fontsize=24,
#                        transform=ax_insights.transAxes)
#        
#        # Title
#        ax_insights.text(x + box_width/2, 0.72, insight['title'],
#                        ha='center', va='top', fontsize=12, fontweight='bold',
#                        transform=ax_insights.transAxes)
#        
#        # Text
#        ax_insights.text(x + box_width/2, 0.45, insight['text'],
#                        ha='center', va='center', fontsize=10,
#                        wrap=True, transform=ax_insights.transAxes,
#                        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
#                                alpha=0.7, edgecolor='none'))
#    
#    # Footer
#    from datetime import datetime
#    footer_text = (f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
#                  f"Total Loans: {m['total_loans']:,} | "
#                  f"Model: XGBoost Classifier")
#    fig.text(0.5, 0.01, footer_text,
#             ha='center', va='bottom', fontsize=9, style='italic',
#             bbox=dict(boxstyle='round,pad=0.5', facecolor=colors['sparamsdary'], 
#                      alpha=0.2, edgecolor='none'))
#    
#    # Log to MLflow if active
#    try:
#        import mlflow
#        if mlflow.active_run():
#            mlflow.log_figure(fig, "enhanced_business_dashboard.png")
#            print("✅ Enhanced business dashboard logged to MLflow")
#    except Exception as e:
#        print(f"Note: MLflow logging skipped - {e}")
#    
#    return fig