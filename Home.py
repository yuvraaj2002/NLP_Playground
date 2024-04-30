import streamlit as st
import plotly.graph_objects as go
import numpy as np
import time

st.markdown(
    """
        <style>
               .block-container {
                    padding-top: 2rem;
                    padding-bottom: 0rem;
                    # padding-left: 2rem;
                    # padding-right:2rem;
                }
                .block-button{
                    padding: 10px; 
                    width: 100%;
                    background-color: #c4fcce;
                }
        </style>
        """,
    unsafe_allow_html=True,
)

def scatter_plot(data_points_cnt, embedding_technique):
    # Generating random data
    np.random.seed(0)
    x = np.random.standard_normal(data_points_cnt)
    y = np.random.standard_normal(data_points_cnt)
    z = np.random.standard_normal(data_points_cnt)

    # Update the position of data points based on embedding technique
    if embedding_technique == 'Bag Of Words':
        x = x * 2
        y = y * 2
        z = z * 2
        # Set the color of the first data point to red and the rest to blue
        colors = ['red'] + ['blue'] * (data_points_cnt - 1)
    elif embedding_technique == 'TF-IDF':
        x = x * 0.5
        y = y * 0.5
        z = z * 0.5
        # Set the color of the second data point to red and the rest to blue
        colors = ['blue', 'red'] + ['blue'] * (data_points_cnt - 2)
    elif embedding_technique == 'Word2Vec':
        x = x * 1.5
        y = y * 1.5
        z = z * 1.5
        # Set the color of the third data point to red and the rest to blue
        colors = ['blue'] * 2 + ['red'] + ['blue'] * (data_points_cnt - 3)
    elif embedding_technique == 'Glove':
        x = x * 0.8
        y = y * 0.8
        z = z * 0.8
        # Set the color of the fourth data point to red and the rest to blue
        colors = ['blue'] * 3 + ['red'] + ['blue'] * (data_points_cnt - 4)
    elif embedding_technique == 'Fasttext':
        x = x * 1.2
        y = y * 1.2
        z = z * 1.2
        # Set the color of the fifth data point to red and the rest to blue
        colors = ['blue'] * 4 + ['red'] + ['blue'] * (data_points_cnt - 5)
    elif embedding_technique == 'ElMo':
        x = x * 1.1
        y = y * 1.1
        z = z * 1.1
        # Set the color of the sixth data point to red and the rest to blue
        colors = ['blue'] * 5 + ['red'] + ['blue'] * (data_points_cnt - 6)
    else:
        x = x * 1.0
        y = y * 1.0
        z = z * 1.0
        # Set the color of the last data point to red and the rest to blue
        colors = ['blue'] * (data_points_cnt - 1) + ['red']

    # Create a trace
    fig = go.Figure(data=[go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=7,
            color=colors,  # set color to list of colors
            colorscale='Viridis',  # choose a colorscale
            opacity=0.8
        )
    )])

    # Update layout
    fig.update_layout(scene=dict(
        xaxis_title='X-axis',
        yaxis_title='Y-axis',
        zaxis_title='Z-axis'),
        margin=dict(l=0, r=0, b=0, t=0),
        title='3D Scatter Plot with Title')

    # Display the plot in Streamlit
    st.plotly_chart(fig)


def plot_histogram():
    # Categories of embedding techniques
    categories = ['Bag Of Words', 'TF-IDF', 'Word2Vec', 'Glove', 'Fasttext', 'ElMo']

    # Custom labels for the horizontal axis
    custom_labels = ['Cosine similarity', 'Word analogy', 'QVEC', 'NLP task']

    # Generating random data for each embedding technique
    np.random.seed(0)
    random_data = {category: np.random.rand(4) for category in categories}

    # Sorting random values for each category
    sorted_data = {category: np.sort(data) for category, data in random_data.items()}

    # Creating the bar plot trace
    fig = go.Figure()
    for category, data in sorted_data.items():
        fig.add_trace(go.Bar(x=custom_labels, y=data, name=category))

    # Updating layout
    fig.update_layout(
        title="Bar Plot with 4 Bars",
        xaxis_title="Category",
        yaxis_title="Value",
        width=800,  # Adjust the width
        height=420  # Adjust the height
    )

    # Displaying the plot in Streamlit
    st.plotly_chart(fig)


def analysis_page():
    Visualization_col, description_col = st.columns(spec=(1,1), gap="large")
    with Visualization_col:
        st.markdown(
            "<h1 style='text-align: left; font-size: 50px; '>Embedding Analysis📊</h1>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='font-size: 19px; text-align: left;'>Word embeddings are dense, low-dimensional representations of words in a continuous vector space, where words with similar meanings or usage patterns are positioned closer together. These embeddings are learned from large corpora of text data using various techniques such as Word2Vec, GloVe, FastText, and ELMo.Overall, by evaluating embeddings, we gain insights into their quality, effectiveness, and suitability for various natural language processing applications, ultimately leading to better-informed decisions when selecting and using embeddings in real-world scenarios.</p>",
            unsafe_allow_html=True,
        )


    with description_col:
        with st.container(border=True):
            st.markdown(
                "<p style='font-size: 19px; text-align: left;background-color: #c4fcce;padding:1rem;'>In this module, the initial step entails the selection of a word embedding technique for which you intend to visualize the dimensionality-reduced 3D scatter plot. Subsequently, you have the option to fine-tune the number of neighboring data points within the corpus. Upon completion, clicking the analyze button triggers the computation of evaluation metric values corresponding to each embedding technique.</p>",
                unsafe_allow_html=True,
            )
            # Create a selectbox
            embedding_option = st.selectbox(
                'Select embedding technique:',
                ('Bag Of Words', 'TF-IDF', 'Word2Vec','Glove','Fasttext','ElMo')
            )

            # Create a slider
            datapoints_cnt = st.slider(
                'How many data points do you want to consider for analysis:',
                min_value=2,
                max_value=350,
                value=200,
                step=1
            )
            if datapoints_cnt and embedding_option is not None:
                with Visualization_col:
                    scatter_plot(datapoints_cnt, embedding_option)

            analyze_bt = st.button("Analayze performance",use_container_width=True)
            if analyze_bt:
                my_bar = st.progress(0, text="Computing values of metrics")

                for percent_complete in range(100):
                    time.sleep(0.01)
                    my_bar.progress(percent_complete + 1, text="Computing values of metrics")
                time.sleep(1)
                my_bar.empty()
                plot_histogram()

analysis_page()
