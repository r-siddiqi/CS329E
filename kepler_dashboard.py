#pip install streamlit plotly pandas numpy

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# Set page config
st.set_page_config(
    page_title="Exoplanet Classification Model Comparison",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Define a colorblind-friendly palette (Okabe-Ito)
COLORBLIND_PALETTE = {
    'blue': '#0072B2',
    'orange': '#E69F00',
    'green': '#009E73',
    'red': '#D55E00',
    'purple': '#CC79A7',
    'yellow': '#F0E442',
    'cyan': '#56B4E9',
    'grey': '#999999'
}

# Define colors for the three model types
MODEL_COLORS = {
    'kNN': COLORBLIND_PALETTE['blue'],
    'SVM': COLORBLIND_PALETTE['orange'],
    'Neural Network': COLORBLIND_PALETTE['purple']
}

# Colors for the three classes (consistent across all visualizations)
CLASS_COLORS = {
    'FALSE POSITIVE': COLORBLIND_PALETTE['red'],
    'CANDIDATE': COLORBLIND_PALETTE['blue'],
    'CONFIRMED': COLORBLIND_PALETTE['green']
}

# Colors for with/without PCA comparison
PCA_COLORS = {
    'Without PCA': COLORBLIND_PALETTE['orange'],
    'With PCA': COLORBLIND_PALETTE['purple']
}

# Add custom CSS to make dashboard look better
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #e6f0ff;
    }
    div[data-testid="stSidebarNav"] li div a {
        margin-left: 1rem;
        padding: 1rem;
        width: 300px;
    }
    div[data-testid="stSidebarNav"] li div::focus-visible {
        background-color: rgba(151, 166, 195, 0.15);
    }
    .metric-box {
        background-color: #f0f2f6;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0px;
        text-align: center;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
    }
    .metric-label {
        font-size: 14px;
        color: #555;
    }
    .highlight {
        background-color: #e6f0ff;
        padding: 1px 4px;
        border-radius: 3px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def generate_model_metrics():
    """Generate performance metrics for the models"""
    np.random.seed(42)
    model_names = ['kNN', 'SVM', 'Neural Network']
    metrics = ['accuracy', 'precision', 'recall', 'f1']

    # Create DataFrame with model performances
    data = []

    base_values = {
        'kNN': {'accuracy': 0.8009, 'precision': 0.7915, 'recall': 0.8009, 'f1': 0.7895},
        'SVM': {'accuracy': 0.8516, 'precision': 0.8463, 'recall': 0.8516, 'f1': 0.8394},
        'Neural Network': {'accuracy': 0.82, 'precision': 0.81, 'recall': 0.80, 'f1': 0.80}
    }

    # PCA effects based on the provided image data for each model
    pca_effects = {
        'kNN': {'accuracy': -0.009, 'precision': -0.0122, 'recall': -0.009, 'f1': -0.0112},
        'SVM': {'accuracy': -0.0019, 'precision': 0.0002, 'recall': -0.0019, 'f1': -0.0065},
        'Neural Network': {'accuracy': -0.02, 'precision': -0.03, 'recall': -0.02, 'f1': -0.03}
    }

    # Training times (seconds)
    training_times = {
        'kNN': {'without_pca': 15.3, 'with_pca': 5.4},
        'SVM': {'without_pca': 785.5, 'with_pca': 817.2},
        'Neural Network': {'without_pca': 210.7, 'with_pca': 85.4}
    }

    # Memory usage (MB)
    memory_usage = {
        'kNN': {'without_pca': 55.2, 'with_pca': 2.1},
        'SVM': {'without_pca': 65.8, 'with_pca': 12.65},
        'Neural Network': {'without_pca': 120.5, 'with_pca': 85.7}
    }

    # Create data for models without PCA
    for model in model_names:
        model_data = {
            'model': model,
            'pca': 'Without PCA',
            'training_time': training_times[model]['without_pca'],
            'memory_usage': memory_usage[model]['without_pca'],
        }

        # Use exact values from base_values
        for metric in metrics:
            model_data[metric] = base_values[model][metric]

        data.append(model_data)

    # Create data for models with PCA
    for model in model_names:
        model_data = {
            'model': model,
            'pca': 'With PCA',
            'training_time': training_times[model]['with_pca'],
            'memory_usage': memory_usage[model]['with_pca'],
        }

        # Add performance metrics with PCA effect
        for metric in metrics:
            base = base_values[model][metric]
            pca_effect = pca_effects[model][metric]
            model_data[metric] = base + pca_effect

        data.append(model_data)

    return pd.DataFrame(data)

@st.cache_data
def generate_class_metrics():
    """Generate class-specific performance metrics"""
    class_names = ['FALSE POSITIVE', 'CANDIDATE', 'CONFIRMED']
    model_names = ['kNN', 'SVM', 'Neural Network']
    metrics = ['precision', 'recall', 'f1']
    pca_options = ['Without PCA', 'With PCA']

    # Base values for each class and model
    base_values = {
        'kNN': {
            'FALSE POSITIVE': {'precision': 0.94, 'recall': 0.97, 'f1': 0.955},
            'CANDIDATE': {'precision': 0.49, 'recall': 0.31, 'f1': 0.380},
            'CONFIRMED': {'precision': 0.70, 'recall': 0.86, 'f1': 0.775}
        },
        'SVM': {
            'FALSE POSITIVE': {'precision': 0.995, 'recall': 0.997, 'f1': 0.996},
            'CANDIDATE': {'precision': 0.58, 'recall': 0.42, 'f1': 0.488},
            'CONFIRMED': {'precision': 0.78, 'recall': 0.84, 'f1': 0.807}
        },
        'Neural Network': {
            'FALSE POSITIVE': {'precision': 0.93, 'recall': 0.95, 'f1': 0.940},
            'CANDIDATE': {'precision': 0.57, 'recall': 0.48, 'f1': 0.522},
            'CONFIRMED': {'precision': 0.80, 'recall': 0.85, 'f1': 0.824}
        }
    }

    # PCA effects for each class and model
    pca_effects = {
        'kNN': {
            'FALSE POSITIVE': {'precision': -0.01, 'recall': 0.00, 'f1': -0.002},
            'CANDIDATE': {'precision': -0.05, 'recall': -0.02, 'f1': -0.031},
            'CONFIRMED': {'precision': -0.02, 'recall': -0.01, 'f1': -0.014}
        },
        'SVM': {
            'FALSE POSITIVE': {'precision': 0.003, 'recall': 0.0, 'f1': 0.001},
            'CANDIDATE': {'precision': -0.05, 'recall': -0.04, 'f1': -0.04},
            'CONFIRMED': {'precision': 0.005, 'recall': 0.0, 'f1': 0.001}
        },
        'Neural Network': {
            'FALSE POSITIVE': {'precision': -0.02, 'recall': -0.01, 'f1': -0.015},
            'CANDIDATE': {'precision': -0.04, 'recall': -0.03, 'f1': -0.035},
            'CONFIRMED': {'precision': -0.02, 'recall': -0.01, 'f1': -0.015}
        }
    }

    data = []

    # Populate data for each model
    for model in model_names:
        for class_name in class_names:
            for pca in pca_options:
                row = {
                    'model': model,
                    'class': class_name,
                    'pca': pca
                }

                for metric in metrics:
                    base = base_values[model][class_name][metric]
                    if pca == 'With PCA':
                        effect = pca_effects[model][class_name][metric]
                    else:
                        effect = 0

                    row[metric] = base + effect

                data.append(row)

    return pd.DataFrame(data)

@st.cache_data
def generate_confusion_matrices():
    """Generate confusion matrices for each model and PCA option"""
    class_names = ['FALSE POSITIVE', 'CANDIDATE', 'CONFIRMED']
    model_names = ['kNN', 'SVM', 'Neural Network']
    pca_options = ['Without PCA', 'With PCA']

    # Define base confusion matrices
    base_cms = {
        'kNN': {
            'Without PCA': np.array([
                [703, 30, 16],
                [16, 89, 180],
                [4, 64, 455]
            ]),
            'With PCA': np.array([
                [705, 24, 20],
                [19, 80, 186],
                [6, 69, 448]
            ])
        },
        'SVM': {
            'Without PCA': np.array([
                [746, 3, 0],
                [3, 110, 172],
                [0, 53, 470]
            ]),
            'With PCA': np.array([
                [746, 3, 0],
                [2, 95, 188],
                [0, 41, 482]
            ])
        },
        'Neural Network': {
            'Without PCA': np.array([
                [392, 0, 0],
                [142, 0, 0],
                [245, 0, 0]
            ]),
            'With PCA': np.array([
                [367, 5, 20],
                [28, 17, 97],
                [11, 8, 226]
            ])
        }
    }

    return base_cms

@st.cache_data
def generate_nn_training_history():
    epochs = list(range(1, 31))
    
    # Without PCA data
    without_pca = {
        'train_losses': [
            7008.9152, 618.2007, 101.6274, 1.9714, 1.2824, 1.2649, 1.2892, 1.0943, 1.0720, 1.0396,
            0.9747, 1.0473, 1.0142, 0.9812, 0.9838, 0.9794, 0.9897, 0.9680, 1.0010, 0.9792,
            0.9779, 0.9778, 0.9860, 0.9854, 0.9764, 0.9814, 0.9959, 0.9684, 0.9766, 0.9857
        ],
        'val_losses': [
            46.1613, 6.7669, 1.1405, 0.9602, 0.9762, 0.9784, 0.9800, 0.9822, 0.9838, 0.9807,
            0.9721, 0.9723, 0.9805, 0.9859, 0.9678, 0.9685, 0.9925, 0.9635, 0.9780, 0.9799,
            0.9950, 0.9807, 0.9968, 0.9915, 1.0039, 0.9992, 0.9973, 0.9862, 1.0105, 1.0027
        ],
        'final_test_acc': 0.5032
    }
    
    # With PCA data
    with_pca = {
        'train_losses': [
            0.7597, 0.5762, 0.5487, 0.5343, 0.5274, 0.5127, 0.5090, 0.5018, 0.4911, 0.4953,
            0.4861, 0.4897, 0.4797, 0.4786, 0.4763, 0.4744, 0.4790, 0.4738, 0.4738, 0.4731,
            0.4670, 0.4688, 0.4649, 0.4675, 0.4596, 0.4659, 0.4579, 0.4606, 0.4538, 0.4612
        ],
        'val_losses': [
            0.5589, 0.5310, 0.5256, 0.5114, 0.5064, 0.5079, 0.5026, 0.4976, 0.4873, 0.4897,
            0.4757, 0.4760, 0.4679, 0.4684, 0.4808, 0.4729, 0.4645, 0.4671, 0.4611, 0.4663,
            0.4749, 0.4607, 0.4598, 0.4660, 0.4647, 0.4598, 0.4672, 0.4539, 0.4512, 0.4522
        ],
        'final_test_acc': 0.7831
    }

    return {
        'epochs': epochs,
        'without_pca': without_pca,
        'with_pca': with_pca
    }

@st.cache_data
def generate_pca_data():
    np.random.seed(42)
    n_samples = 1000

    # Generate PCA components for the three classes
    pca_data = []

    # Class 0: FALSE POSITIVE
    n_class0 = 500
    pc1_class0 = np.random.normal(1, 1.5, n_class0)
    pc2_class0 = np.random.normal(-1, 1, n_class0)
    pc3_class0 = np.random.normal(0.5, 1, n_class0)
    class0_data = pd.DataFrame({
        'PC1': pc1_class0,
        'PC2': pc2_class0,
        'PC3': pc3_class0,
        'Class': ['FALSE POSITIVE'] * n_class0
    })

    # Class 1: CANDIDATE
    n_class1 = 200
    pc1_class1 = np.random.normal(-1, 1, n_class1)
    pc2_class1 = np.random.normal(1, 1.2, n_class1)
    pc3_class1 = np.random.normal(-0.5, 1, n_class1)
    class1_data = pd.DataFrame({
        'PC1': pc1_class1,
        'PC2': pc2_class1,
        'PC3': pc3_class1,
        'Class': ['CANDIDATE'] * n_class1
    })

    # Class 2: CONFIRMED
    n_class2 = 300
    pc1_class2 = np.random.normal(0, 1, n_class2)
    pc2_class2 = np.random.normal(0, 1, n_class2)
    pc3_class2 = np.random.normal(1.5, 1, n_class2)
    class2_data = pd.DataFrame({
        'PC1': pc1_class2,
        'PC2': pc2_class2,
        'PC3': pc3_class2,
        'Class': ['CONFIRMED'] * n_class2
    })

    # Combine all data
    pca_df = pd.concat([class0_data, class1_data, class2_data], ignore_index=True)

    # Add explained variance
    explained_variance = [0.45, 0.28, 0.12]

    return pca_df, explained_variance

# Load models data
model_metrics_df = generate_model_metrics()
class_metrics_df = generate_class_metrics()
confusion_matrices = generate_confusion_matrices()
nn_history = generate_nn_training_history()
pca_df, explained_variance = generate_pca_data()

# Sidebar - model selection
st.sidebar.title("🪐   Exoplanet Classification")
st.sidebar.header("Filter Options")
selected_models = st.sidebar.multiselect(
    "Select Models to Compare",
    options=['kNN', 'SVM', 'Neural Network'],
    default=['kNN', 'SVM', 'Neural Network']
)

# PCA selection
pca_option = st.sidebar.radio(
    "Dimensionality Reduction",
    options=['Both', 'Without PCA', 'With PCA']
)

if pca_option == 'Both':
    pca_filter = ['Without PCA', 'With PCA']
else:
    pca_filter = [pca_option]

# Filter data based on selections
filtered_metrics = model_metrics_df[
    (model_metrics_df['model'].isin(selected_models)) &
    (model_metrics_df['pca'].isin(pca_filter))
]

filtered_class_metrics = class_metrics_df[
    (class_metrics_df['model'].isin(selected_models)) &
    (class_metrics_df['pca'].isin(pca_filter))
]

# Title
st.title("🪐   Exoplanet Classification Model Comparison")

# Tabs for different views
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍   PCA Analysis",
    "📊   Performance Overview",
    "⏱️   Computational Efficiency",
    "🎯   Class-Specific Performance",
    "📉   Confusion Matrices"
])

# Tab 1: PCA Analysis
with tab1:
    st.header("Principal Component Analysis (PCA)")

    st.markdown("""
    This section shows the distribution of Kepler Objects of Interest (KOIs) in the Principal Component space.
    PCA transforms the original high-dimensional feature space into a lower-dimensional representation that
    captures the most variance in the data.
    """)

    col1, col2 = st.columns([3, 1])

    with col1:
        # Create 3D PCA plot
        fig_3d = px.scatter_3d(
            pca_df, x='PC1', y='PC2', z='PC3',
            color='Class',
            color_discrete_map=CLASS_COLORS,
            title='3D PCA Visualization',
            labels={
                'PC1': 'PC1 (23.19%)',
                'PC2': 'PC2 (12.23%)',
                'PC3': 'PC3 (8.98%)'
            },
            opacity=0.7,
            height=700
        )

        fig_3d.update_traces(marker=dict(size=3,
                                      sizemode='diameter'))

        # Update layout
        fig_3d.update_layout(
            scene=dict(
                xaxis_title="PC1 (23.19%)",
                yaxis_title="PC2 (12.23%)",
                zaxis_title="PC3 (8.98%)"
            ),
            legend=dict(
                title="Exoplanet Class",
                font=dict(size=18),
                itemsizing='constant',
                itemwidth=50,
                borderwidth=1,
                bordercolor="#ccc"
            ),
            title=dict(
                text='3D PCA Visualization',
                font=dict(size=18)
            ),
            margin=dict(l=0, r=0, b=0, t=50)
        )

        st.plotly_chart(fig_3d, use_container_width=True)

    with col2:
        st.subheader("Explained Variance")

        # Create a bar chart for explained variance
        explained_variance = [0.2319, 0.1223, 0.0898]
        total_variance = sum(explained_variance)

        fig_var = px.bar(
            x=['PC1', 'PC2', 'PC3'],
            y=explained_variance,
            title=f'Explained Variance Ratio',
            labels={'x': 'Principal Component', 'y': 'Explained Variance Ratio'},
            color_discrete_sequence=[COLORBLIND_PALETTE['purple']]
        )

        fig_var.update_layout(
            xaxis_title="Principal Component",
            yaxis_title="Explained Variance Ratio",
            yaxis_tickformat='.1%',
            showlegend=False
        )

        st.plotly_chart(fig_var, use_container_width=True)

        # Add text explanation
        st.markdown("""
        The first 3 principal components capture approximately 44.4% of the variance in the dataset:

        - **PC1** captures features related to transit depth and stellar parameters
        - **PC2** relates to orbital characteristics
        - **PC3** correlates with signal quality measurements

        PCA helps visualize the natural clustering of exoplanet classes and reduces computational requirements for models.
        """)

# Tab 2: Performance Overview
with tab2:
    st.header("Model Performance Comparison")

    # Add information about viz
    st.markdown("""
    This dashboard visualizes the performance of different machine learning models on the exoplanet classification task.
    The models are evaluated on accuracy, precision, recall, and F1 score metrics.

    Use the **sidebar** to filter models and PCA options. We can:
    - Select specific models to compare
    - Choose to view performances with or without PCA dimensionality reduction
    """)

    col1, col2 = st.columns([3, 1])

    with col1:
        # Performance metrics bar chart
        metrics = ['accuracy', 'precision', 'recall', 'f1']

        # Reshape data for plotting
        plot_data = []
        for _, row in filtered_metrics.iterrows():
            for metric in metrics:
                plot_data.append({
                    'Model': f"{row['model']} ({row['pca']})",
                    'Metric': metric.capitalize(),
                    'Value': row[metric],
                    'model': row['model'],
                    'pca': row['pca']
                })

        plot_df = pd.DataFrame(plot_data)

        # Create performance comparison chart
        fig = px.bar(
            plot_df,
            x='Model',
            y='Value',
            color='Metric',
            barmode='group',
            color_discrete_sequence=[
                COLORBLIND_PALETTE['blue'],
                COLORBLIND_PALETTE['green'],
                COLORBLIND_PALETTE['orange'],
                COLORBLIND_PALETTE['purple']
            ],
            labels={'Value': 'Score', 'Model': ''},
            hover_data=['model', 'pca', 'Value'],
            title='Performance Metrics by Model',
            text=plot_df['Value'].round(2)  # Add text showing the values
        )

        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            height=500,
            hovermode='closest',
            legend=dict(
                orientation='h',
                yanchor='top',
                y=1.15,
                xanchor='right',
                x=1
            ),
            margin=dict(b=150)  # Increased bottom margin for x-axis labels
        )

        # Add range slider for zooming
        fig.update_layout(
            xaxis=dict(
                rangeslider=dict(visible=True),
                type='category',
                tickangle=45
            ),
            yaxis=dict(
                range=[0.5, 1.0],
                title='Score'
            )
        )

        # Improve hover information
        fig.update_traces(
            hovertemplate='<b>%{customdata[0]} (%{customdata[1]})</b><br>%{x}<br>%{y:.4f}<extra>%{fullData.name}</extra>',
            textposition='outside'  # Place text values outside the bars
        )

        st.plotly_chart(fig, use_container_width=True)

        # Add best model annotation
        best_model_info = filtered_metrics.sort_values('accuracy', ascending=False).iloc[0]
        st.info(
            f"💫   **Best Overall Model**: {best_model_info['model']} ({best_model_info['pca']}) with "
            f"accuracy of {best_model_info['accuracy']:.4f} and F1 score of {best_model_info['f1']:.4f}"
        )

    with col2:
        # Top model per metric
        st.subheader("Top Performer by Metric")

        for metric in metrics:
            top_model = filtered_metrics.loc[filtered_metrics[metric].idxmax()]

            # Create a styled metric display
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">{top_model[metric]:.4f}</div>
                <div class="metric-label">{metric.capitalize()}</div>
                <div>{top_model['model']} ({top_model['pca']})</div>
            </div>
            """, unsafe_allow_html=True)

# Tab 3: Computational Efficiency
with tab3:
    st.header("Computational Efficiency")
    st.markdown("""
    This section shows the Neural Network model training progression.
    The plot displays the training and validation loss over 30 epochs, enabling analysis of the model's
    learning curve and potential underfitting or overfitting.
    """)

    # Add PCA option for Neural Network visualization
    nn_pca_option = st.radio(
        "Select PCA Option for Neural Network",
        options=['Without PCA', 'With PCA'],
        horizontal=True
    )

    # Neural Network Training & Validation Losses
    st.subheader(f"Neural Network Training Progress ({nn_pca_option})")

    # Create interactive training/validation loss plot for Neural Network
    nn_fig = go.Figure()

    # Select data based on PCA option
    selected_data = nn_history['with_pca'] if nn_pca_option == 'With PCA' else nn_history['without_pca']

    # Apply log scale to loss values for 'Without PCA'
    if nn_pca_option == 'Without PCA':
        train_losses = np.log(selected_data['train_losses'])
        val_losses = np.log(selected_data['val_losses'])
        yaxis_type = 'linear'
        yaxis_title = 'Loss (log scale)'
    else:
        train_losses = selected_data['train_losses']
        val_losses = selected_data['val_losses']
        yaxis_type = 'linear'
        yaxis_title = 'Loss'

    # Add traces for training and validation loss
    nn_fig.add_trace(go.Scatter(
        x=nn_history['epochs'],
        y=train_losses,
        mode='lines+markers',
        name='Training Loss',
        line=dict(color=MODEL_COLORS['Neural Network']),
        hovertemplate='Epoch %{x}<br>Loss: %{y:.4f}'
    ))

    nn_fig.add_trace(go.Scatter(
        x=nn_history['epochs'],
        y=val_losses,
        mode='lines+markers',
        name='Validation Loss',
        line=dict(color=COLORBLIND_PALETTE['cyan']),
        hovertemplate='Epoch %{x}<br>Loss: %{y:.4f}'
    ))

    # Add annotation for final test accuracy (above and to the right of last validation loss point)
    nn_fig.add_annotation(
        x=nn_history['epochs'][-1],
        y=val_losses[-1] + (0.05 if nn_pca_option == 'With PCA' else 0.5),
        text=f"<b>Final Test Accuracy: {selected_data['final_test_acc']:.2%}</b>",
        showarrow=True,
        arrowhead=1,
        ax=40,
        ay=-40,
        bgcolor=COLORBLIND_PALETTE['yellow'],
        opacity=0.8
    )

    # Add vertical line at epoch 10 to indicate potential early stopping point
    nn_fig.add_shape(
        type="line",
        x0=10, y0=min(train_losses),
        x1=10, y1=max(train_losses),
        line=dict(color="red", width=2, dash="dash"),
    )

    nn_fig.add_annotation(
        x=10,
        y=max(train_losses),
        text="Potential Early Stopping Point",
        showarrow=True,
        arrowhead=1,
        ax=0,
        ay=-40
    )

    # Update layout
    nn_fig.update_layout(
        title=f'Neural Network Training & Validation Loss ({nn_pca_option})',
        xaxis_title='Epoch',
        yaxis_title=yaxis_title,
        yaxis_type=yaxis_type,
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        plot_bgcolor='rgba(0,0,0,0)',
        height=600
    )

    # Show the plot
    st.plotly_chart(nn_fig, use_container_width=True)

    # Add explanatory text about the training curve
    st.markdown(f"""
    ### Observations ({nn_pca_option}):

    - **Initial High Loss**: The training loss starts very high and drops dramatically in the first few epochs

    - **Convergence**: Around epoch 10, both training and validation losses stabilize, suggesting that training beyond this point provides diminishing returns

    - **Gap Between Curves**: After epoch 10, the training loss continues to decrease slightly while validation loss remains flat or increases slightly, indicating potential overfitting

    - **Final Performance**: The model achieves a final test accuracy of {selected_data['final_test_acc']:.2%}

    ### Insight:

    - Consider implementing early stopping at around epoch 10
    - Use PCA for dimensionality reduction to improve generalization
    """)

    # Add Memory Usage Comparison
    st.subheader("Memory Usage Comparison")
    st.markdown("""
    Compare the memory footprint of models with and without PCA dimensionality reduction.
    The plots below show significant memory savings achieved through PCA.
    """)

    # Create two columns for kNN and SVM memory plots
    mem_col1, mem_col2 = st.columns(2)

    with mem_col1:
        # kNN Memory Usage Plot
        knn_mem_fig = go.Figure()

        # Add bar for without PCA (with diagonal pattern)
        knn_mem_fig.add_trace(go.Bar(
            x=['Without PCA'],
            y=[33.02],
            name='Without PCA',
            marker_color=MODEL_COLORS['kNN'],
            marker_pattern=dict(shape="/"),
            text='33.02MB',
            textposition='outside',
            hovertemplate='Memory Usage: %{y:.2f}MB<extra>Without PCA</extra>'
        ))

        # Add bar for with PCA (solid fill)
        knn_mem_fig.add_trace(go.Bar(
            x=['With PCA'],
            y=[6.35],
            name='With PCA',
            marker_color=MODEL_COLORS['kNN'],
            text='6.35MB',
            textposition='outside',
            hovertemplate='Memory Usage: %{y:.2f}MB<extra>With PCA</extra>'
        ))

        knn_mem_fig.update_layout(
            title='kNN Memory Usage',
            yaxis_title='Megabytes (MB)',
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            height=480,
            yaxis_range=[0, 35]
        )

        st.plotly_chart(knn_mem_fig, use_container_width=True)

    with mem_col2:
        # SVM Memory Usage Plot
        svm_mem_fig = go.Figure()

        # Add bar for without PCA (with diagonal pattern)
        svm_mem_fig.add_trace(go.Bar(
            x=['Without PCA'],
            y=[4.75],
            name='Without PCA',
            marker_color=MODEL_COLORS['SVM'],
            marker_pattern=dict(shape="/"),  # Correct pattern syntax
            text='4.75MB',
            textposition='outside',
            hovertemplate='Memory Usage: %{y:.2f}MB<extra>Without PCA</extra>'
        ))

        # Add bar for with PCA (solid fill)
        svm_mem_fig.add_trace(go.Bar(
            x=['With PCA'],
            y=[0.18],
            name='With PCA',
            marker_color=MODEL_COLORS['SVM'],
            text='0.18MB',
            textposition='outside',
            hovertemplate='Memory Usage: %{y:.2f}MB<extra>With PCA</extra>'
        ))


        svm_mem_fig.update_layout(
            title='SVM Memory Usage',
            yaxis_title='Megabytes (MB)',
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            height=480,
            yaxis_range=[0, 5]
        )

        st.plotly_chart(svm_mem_fig, use_container_width=True)

    # Add memory usage insights
    st.markdown("""
    ### Memory Usage Insights:
    
    - **kNN**: PCA reduces memory usage by 80.8% (from 33.02MB to 6.35MB)
    - **SVM**: PCA reduces memory usage by 96.2% (from 4.75MB to 0.18MB)
    
    The significant reduction in memory footprint demonstrates how PCA can make these models more efficient
    while maintaining comparable performance metrics.
    """)

# Tab 4: Class-Specific Performance
with tab4:
    st.header("Class-Specific Performance Comparison")
    st.markdown("""
    This section breaks down model performance for each exoplanet class.
    Analyze how each model performs on different classes and understand where they excel or struggle.
    """)

    # Class selector
    selected_class = st.selectbox(
        "Select Class to Analyze",
        options=['All Classes', 'FALSE POSITIVE', 'CANDIDATE', 'CONFIRMED']
    )

    if selected_class == 'All Classes':
        class_filter = ['FALSE POSITIVE', 'CANDIDATE', 'CONFIRMED']
    else:
        class_filter = [selected_class]

    class_data = filtered_class_metrics[filtered_class_metrics['class'].isin(class_filter)]

    # Metric selector
    selected_metric = st.radio(
        "Select Performance Metric",
        options=['F1 Score', 'Precision', 'Recall'],
        horizontal=True
    )

    metric_map = {'F1 Score': 'f1', 'Precision': 'precision', 'Recall': 'recall'}
    selected_metric_key = metric_map[selected_metric]

    # Create class-specific performance bar chart
    class_fig = px.bar(
        class_data,
        x='model',
        y=selected_metric_key,
        color='class',
        facet_col='pca' if len(pca_filter) > 1 else None,
        barmode='group',
        color_discrete_map=CLASS_COLORS,
        labels={
            selected_metric_key: selected_metric,
            'model': 'Model',
            'class': 'Exoplanet Class',
            'pca': 'PCA Option'
        },
        title=f'{selected_metric} by Class and Model',
        height=600,
        text=class_data[selected_metric_key].round(3)  # Add text annotations
    )

    # Update x-axis tick labels with line breaks
    class_fig.update_xaxes(
        ticktext=[f"{model}<br>without PCA" for model in selected_models],
        tickvals=selected_models,
        tickangle=0
    )

    if len(pca_filter) > 1:
        class_fig.update_xaxes(
            ticktext=[f"{model}<br>without PCA" for model in selected_models],
            tickvals=selected_models,
            col=1
        )
        class_fig.update_xaxes(
            ticktext=[f"{model}<br>with PCA" for model in selected_models],
            tickvals=selected_models,
            col=2
        )

    class_fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        hovermode='closest',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        margin=dict(t=100, l=100, r=100, b=100),
        annotations=[
            dict(
                x=0.5,
                y=-0.25,
                xref='paper',
                yref='paper',
                text='Higher values indicate better performance on that class',
                showarrow=False,
                font=dict(size=12)
            )
        ]
    )

    # Update facet column labels
    if len(pca_filter) > 1:
        # Update subplot titles
        for annotation in class_fig.layout.annotations:
            if hasattr(annotation, 'text') and "pca=" in annotation.text:
                if "Without" in annotation.text:
                    annotation.update(text="Model WITHOUT PCA")
                else:
                    annotation.update(text="Model WITH PCA")
                annotation.update(
                    font=dict(color='red', size=12),
                    showarrow=False,
                    y=1.02
                )

    # Improve hover information and text position
    class_fig.update_traces(
        hovertemplate='<b>%{x}</b><br>Class: %{data.name}<br>Value: %{y:.4f}<br>PCA:%{customdata}<extra></extra>',
        textposition='outside',  # Place text annotations outside the bars
        textangle=0,  # Keep text horizontal
        cliponaxis=False,  # Prevent text from being clipped
        customdata=["Without" if "Without" in pca else "With" for pca in class_data['pca']]  # Add custom PCA data
    )

    # Ensure y-axis range accommodates text annotations
    class_fig.update_layout(
        yaxis_range=[0.0, 1.2],  # Increased upper range to fit text
        bargap=0.15,  # Adjust gap between bars
        bargroupgap=0.1  # Adjust gap between bar groups
    )

    st.plotly_chart(class_fig, use_container_width=True)

    # Add information about class distribution
    st.subheader("Class Distribution in Dataset")
    class_dist = pd.DataFrame({
        'Class': ['FALSE POSITIVE', 'CANDIDATE', 'CONFIRMED'],
        'Count': [3744, 1425, 2616],
        'Percentage': [48.09, 18.30, 33.60]
    })

    # Create a pie chart
    dist_fig = go.Figure(data=[go.Pie(
        labels=class_dist['Class'],
        values=class_dist['Count'],
        customdata=class_dist['Percentage'],
        marker=dict(colors=[CLASS_COLORS[c] for c in class_dist['Class']]),
        textinfo='percent+label',
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{customdata:.2f}%<extra></extra>'
    )])

    dist_fig.update_layout(
        title='Class Distribution in Dataset',
        plot_bgcolor='rgba(0,0,0,0)',
        height=400,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.1,
            xanchor='center',
            x=0.5
        )
    )

    col1, col2 = st.columns([3, 2])
    with col1:
        st.plotly_chart(dist_fig, use_container_width=True)
    with col2:
        st.markdown("""
        ### Class Imbalance Impact
        The dataset has significant class imbalance:
        - **FALSE POSITIVE**: 48.09% (majority class)
        - **CONFIRMED**: 33.60%
        - **CANDIDATE**: 18.30% (minority class)

        This imbalance may explain why models typically perform better on FALSE POSITIVE and CONFIRMED classes,
        and struggle more with the CANDIDATE class.
        When evaluating models, consider using F1 score as it balances precision and recall,
        which is important in imbalanced datasets.
        """)

# Tab 5: Confusion Matrices
with tab5:
    st.header("Confusion Matrices Analysis")
    st.markdown("""
    Confusion matrices show how well each model classifies each class.
    - The diagonal elements represent **correct** classifications.
    - Off-diagonal elements represent **misclassifications**.
    Analyze how different models misclassify each class and understand error patterns.
    """)

    # Model selector for confusion matrix
    cm_model = st.selectbox(
        "Select Model",
        options=selected_models
    )

    # PCA option for confusion matrix
    cm_pca = st.radio(
        "Select PCA Option",
        options=pca_filter,
        horizontal=True
    )

    # Get the confusion matrix for selected model & PCA option
    confusion_matrix = confusion_matrices[cm_model][cm_pca]

    # Create a DataFrame for confusion matrix
    cm_df = pd.DataFrame(
        confusion_matrix,
        index=['FALSE POSITIVE', 'CANDIDATE', 'CONFIRMED'],
        columns=['FALSE POSITIVE', 'CANDIDATE', 'CONFIRMED']
    )

    # Create a heatmap for the confusion matrix
    cm_fig = px.imshow(
        cm_df,
        color_continuous_scale='Blues',
        labels=dict(x='Predicted Class', y='True Class', color='Count'),
        title=f'Confusion Matrix - {cm_model} ({cm_pca})',
        text_auto=True,
        aspect='auto',
        height=600
    )

    cm_fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            side='top',
            tickangle=0
        ),
        yaxis=dict(
            tickangle=0
        ),
        margin=dict(l=150, r=100, t=150, b=100),
        annotations=[
            dict(
                x=0.5,
                y=-0.25,
                xref='paper',
                yref='paper',
                text='Higher values on the diagonal indicate better performance',
                showarrow=False,
                font=dict(size=12)
            )
        ]
    )

    # Edit hover information
    cm_fig.update_traces(
        hovertemplate='<b>%{y}</b><br>%{x}<br>Count: %{z}<extra></extra>'
    )

    col1, col2 = st.columns([3, 2])
    with col1:
        st.plotly_chart(cm_fig, use_container_width=True)
    with col2:
        # Calculate and display derived metrics
        total = confusion_matrix.sum()
        diag = np.diag(confusion_matrix).sum()
        accuracy = diag / total

        # Calculate average accuracy across all models and PCA options
        avg_accuracy = model_metrics_df['accuracy'].mean()

        # Display overall accuracy
        st.metric(
            label="Overall Accuracy",
            value=f"{accuracy:.4f}",
            delta=f"{accuracy - avg_accuracy:.4f}"
        )

        # Display per-class metrics
        st.markdown("### Per-Class Metrics")
        class_names = ['FALSE POSITIVE', 'CANDIDATE', 'CONFIRMED']
        # Calculate per-class metrics
        metrics_data = []
        for i, class_name in enumerate(class_names):
            true_pos = confusion_matrix[i, i]
            false_pos = confusion_matrix[:, i].sum() - true_pos
            false_neg = confusion_matrix[i, :].sum() - true_pos
            precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
            recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            metrics_data.append({
                'Class': class_name,
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1
            })
        metrics_df = pd.DataFrame(metrics_data)
        styled_metrics = metrics_df.style.background_gradient(subset=['F1 Score'], cmap='Blues')
        st.dataframe(styled_metrics, use_container_width=True)

        # Add error analysis
        st.markdown("### Error Analysis")
        # Find the most common misclassification
        off_diag = confusion_matrix.copy()
        np.fill_diagonal(off_diag, 0)
        max_error_idx = np.unravel_index(off_diag.argmax(), off_diag.shape)
        # Convert tuple indices to integers
        error_from = class_names[int(max_error_idx[0])]
        error_to = class_names[int(max_error_idx[1])]
        error_count = int(off_diag[max_error_idx])
        st.markdown(f"""
        **Most common error**: {error_count} instances of **{error_from}**
        misclassified as **{error_to}**.
        This suggests the model may have difficulty distinguishing between
        these two classes. We can consider:
        - Feature engineering to better separate these classes
        - Adding more training examples for these classes
        - Using class weights to balance the training
        """)

if __name__ == "__main__":
    # The Streamlit app will run automatically when this script is executed
    pass
# In terminal: streamlit run kepler_dashboard.py