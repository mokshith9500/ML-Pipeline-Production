"""
dashboard.py
------------
Interactive HTML Dashboard Generator for the ML Pipeline.

Generates a self-contained HTML file with:
  - Pipeline run summary
  - Model comparison charts
  - Data quality metrics
  - SHAP feature importance visualization
  - Drift detection results

The dashboard is auto-generated after each successful pipeline run.
"""

import json
import os
import base64
from datetime import datetime
from typing import Any, Dict, Optional

def _load_file_as_base64(filepath: str) -> Optional[str]:
    """Load an image file and encode it as base64 for embedding in HTML."""
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')
    except:
        return None


def generate_dashboard(
    validation_report: Dict[str, Any],
    training_results: Dict[str, Any],
    output_path: str = "models/dashboard.html"
) -> str:
    """
    Generate an interactive HTML dashboard.
    
    Args:
        validation_report: The validation report from validation.py
        training_results: The training results from training.py
        output_path: Where to save the HTML file
    
    Returns:
        Path to the generated dashboard
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load SHAP plot if available
    shap_plot_b64 = None
    if training_results.get("shap_plot_path"):
        shap_plot_b64 = _load_file_as_base64(training_results["shap_plot_path"])
    
    # Extract metrics
    best_model = training_results.get("best_model_name", "unknown")
    metrics = training_results.get("metrics", {})
    all_metrics = training_results.get("all_model_metrics", {})
    
    val_status = validation_report.get("status", "UNKNOWN")
    val_metrics = validation_report.get("metrics", {})
    
    # Build model comparison data for charts
    model_names = list(all_metrics.keys())
    model_roc_aucs = [all_metrics[m].get("roc_auc", 0) for m in model_names]
    model_accuracies = [all_metrics[m].get("accuracy", 0) for m in model_names]
    
    # Generate HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ML Pipeline Dashboard — Binary Classification</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            color: #333;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        
        .header {{
            background: white;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            text-align: center;
        }}
        
        .header h1 {{
            color: #667eea;
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        
        .header p {{
            color: #666;
            font-size: 1.1em;
        }}
        
        .status-badge {{
            display: inline-block;
            padding: 8px 20px;
            border-radius: 20px;
            font-weight: 600;
            margin-top: 15px;
            font-size: 1.1em;
        }}
        
        .status-pass {{
            background: #d4edda;
            color: #155724;
        }}
        
        .status-fail {{
            background: #f8d7da;
            color: #721c24;
        }}
        
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}
        
        .card {{
            background: white;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        }}
        
        .card h2 {{
            color: #667eea;
            margin-bottom: 20px;
            font-size: 1.4em;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        
        .metric {{
            display: flex;
            justify-content: space-between;
            padding: 12px 0;
            border-bottom: 1px solid #eee;
        }}
        
        .metric:last-child {{
            border-bottom: none;
        }}
        
        .metric-label {{
            font-weight: 500;
            color: #666;
        }}
        
        .metric-value {{
            font-weight: 700;
            color: #333;
            font-size: 1.1em;
        }}
        
        .metric-value.highlight {{
            color: #667eea;
        }}
        
        .chart-container {{
            position: relative;
            height: 300px;
            margin-top: 20px;
        }}
        
        .full-width {{
            grid-column: 1 / -1;
        }}
        
        .shap-img {{
            width: 100%;
            max-width: 800px;
            margin: 20px auto;
            display: block;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        .winner-badge {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.85em;
            margin-left: 10px;
        }}
        
        .footer {{
            text-align: center;
            color: white;
            margin-top: 30px;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏥 ML Pipeline Dashboard</h1>
            <p>Binary Classification | Real Medical Data (Pima Indians Dataset)</p>
            <span class="status-badge status-{val_status.lower()}">{val_status}</span>
            <p style="margin-top: 15px; color: #999;">
                Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            </p>
        </div>
        
        <div class="grid">
            <!-- Best Model Card -->
            <div class="card">
                <h2>🏆 Best Model</h2>
                <div class="metric">
                    <span class="metric-label">Model</span>
                    <span class="metric-value highlight">{best_model.upper()}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">ROC-AUC</span>
                    <span class="metric-value">{metrics.get('roc_auc', 0):.4f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Accuracy</span>
                    <span class="metric-value">{metrics.get('accuracy', 0)*100:.1f}%</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Precision</span>
                    <span class="metric-value">{metrics.get('precision', 0):.4f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Recall</span>
                    <span class="metric-value">{metrics.get('recall', 0):.4f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">F1 Score</span>
                    <span class="metric-value">{metrics.get('f1_score', 0):.4f}</span>
                </div>
            </div>
            
            <!-- Data Quality Card -->
            <div class="card">
                <h2>📊 Data Quality</h2>
                <div class="metric">
                    <span class="metric-label">Total Rows</span>
                    <span class="metric-value">{val_metrics.get('total_rows', 0):,}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Diabetes Rate</span>
                    <span class="metric-value">{val_metrics.get('diabetes_rate', 0)*100:.1f}%</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Missing Values</span>
                    <span class="metric-value">{val_metrics.get('overall_missing_pct', 0)*100:.2f}%</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Duplicates</span>
                    <span class="metric-value">{val_metrics.get('duplicate_rows', 0)}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Validation Errors</span>
                    <span class="metric-value">{len(validation_report.get('errors', []))}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Validation Warnings</span>
                    <span class="metric-value">{len(validation_report.get('warnings', []))}</span>
                </div>
            </div>
            
            <!-- Model Comparison Chart -->
            <div class="card full-width">
                <h2>📈 Model Comparison — ROC-AUC</h2>
                <div class="chart-container">
                    <canvas id="rocChart"></canvas>
                </div>
            </div>
            
            <!-- Accuracy Comparison Chart -->
            <div class="card full-width">
                <h2>📉 Model Comparison — Accuracy</h2>
                <div class="chart-container">
                    <canvas id="accChart"></canvas>
                </div>
            </div>
            
            <!-- SHAP Explainability -->
            {f'''
            <div class="card full-width">
                <h2>🔍 SHAP Feature Importance</h2>
                <p style="color: #666; margin-bottom: 15px;">
                    Shows which features drive the model's predictions. Critical for healthcare applications.
                </p>
                <img src="data:image/png;base64,{shap_plot_b64}" class="shap-img" alt="SHAP Plot">
            </div>
            ''' if shap_plot_b64 else ''}
        </div>
        
        <div class="footer">
            <p>ML Pipeline | Production-Grade Binary Classification System</p>
            <p style="margin-top: 5px;">Built with Python, scikit-learn, XGBoost, LightGBM, SHAP</p>
        </div>
    </div>
    
    <script>
        // Model names and data
        const modelNames = {json.dumps(model_names)};
        const modelROCAUCs = {json.dumps(model_roc_aucs)};
        const modelAccuracies = {json.dumps(model_accuracies)};
        const bestModel = "{best_model}";
        
        // ROC-AUC Chart
        const rocCtx = document.getElementById('rocChart').getContext('2d');
        const rocColors = modelNames.map(m => 
            m === bestModel ? 'rgba(102, 126, 234, 0.8)' : 'rgba(102, 126, 234, 0.3)'
        );
        const rocBorders = modelNames.map(m => 
            m === bestModel ? 'rgba(102, 126, 234, 1)' : 'rgba(102, 126, 234, 0.5)'
        );
        
        new Chart(rocCtx, {{
            type: 'bar',
            data: {{
                labels: modelNames,
                datasets: [{{
                    label: 'ROC-AUC Score',
                    data: modelROCAUCs,
                    backgroundColor: rocColors,
                    borderColor: rocBorders,
                    borderWidth: 2
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ display: false }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true,
                        max: 1.0
                    }}
                }}
            }}
        }});
        
        // Accuracy Chart
        const accCtx = document.getElementById('accChart').getContext('2d');
        const accColors = modelNames.map(m => 
            m === bestModel ? 'rgba(118, 75, 162, 0.8)' : 'rgba(118, 75, 162, 0.3)'
        );
        const accBorders = modelNames.map(m => 
            m === bestModel ? 'rgba(118, 75, 162, 1)' : 'rgba(118, 75, 162, 0.5)'
        );
        
        new Chart(accCtx, {{
            type: 'bar',
            data: {{
                labels: modelNames,
                datasets: [{{
                    label: 'Accuracy',
                    data: modelAccuracies,
                    backgroundColor: accColors,
                    borderColor: accBorders,
                    borderWidth: 2
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ display: false }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true,
                        max: 1.0
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>"""
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    
    return output_path
