#----------------------------- Libraries ------------------------------#
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
import plotly.graph_objects as go

#----------------------------- Spark Session ------------------------------#
spark = SparkSession.builder \
        .appName("ResearchDataProcessing") \
        .getOrCreate()

#----------------------------- Data Preprocessing ------------------------------#
df_spark = spark.read.option("header", "true").csv("data.csv")
df = df_spark.toPandas()

#----------------------------- Data Cleaning ------------------------------#
df['keywords'] = df['keywords'].fillna('').str.split(',')
df['subjects'] = df['subjects'].fillna('').str.split(',')
df['author_full_name'] = df['author_full_name'].fillna('').str.split(',')
df['affiliation_ids'] = df['affiliation_ids'].fillna('').str.split(',')
df['keywords'] = df['keywords'].apply(lambda x: [kw.strip() for kw in x if kw.strip()])
df['subjects'] = df['subjects'].apply(lambda x: [subj.strip() for subj in x if subj.strip()])
df['author_full_name'] = df['author_full_name'].apply(lambda x: [author.strip() for author in x if author.strip()])
df['affiliation_ids'] = df['affiliation_ids'].apply(lambda x: [aff.strip() for aff in x if aff.strip()])
df['title'] = df['title'].fillna('') 

#----------------------------- Sidebar ------------------------------#
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ("Search", "Statistics"))


############################################### SEARCH ###############################################################


if page == "Search":
    st.title("Scopus Paper Similarity Detector")

    #----------------------------- Search Bar ------------------------------#
    search_query = st.text_input("Search by Title, Author, Keywords, or Subjects :")
    if search_query:
        filtered_df = df[
            df['title'].str.contains(search_query, case=False) | 
            df['author_full_name'].str.contains(search_query, case=False) | 
            df['keywords'].apply(lambda x: any(search_query.lower() in keyword.lower() for keyword in x)) |
            df['subjects'].str.contains(search_query, case=False)
        ]
    else:
        filtered_df = df
    
    if not filtered_df.empty:
        #----------------------------- Select Box ------------------------------#
        selected_paper = st.selectbox("Select a paper to find similar papers :", filtered_df['title'])
        st.markdown("<hr>", unsafe_allow_html=True)

        #----------------------------- Display Paper Info ------------------------------#
        selected_paper_row = filtered_df[filtered_df['title'] == selected_paper].iloc[0]
        st.write(f"### **{selected_paper_row['title']}**")
        st.write(f"Author : **{', '.join(selected_paper_row['author_full_name'])}**")
        st.write(f"Abstract : {selected_paper_row['abstract']}")
        keywords_text = ", ".join(selected_paper_row['keywords'])
        st.write(f"Keywords: {keywords_text}")
        subject_text = ", ".join(selected_paper_row['subjects'])
        st.write(f"Subjects : {subject_text}")
        st.markdown("<hr>", unsafe_allow_html=True)

        #----------------------------- Display Top 5 Similar Papers ------------------------------#
        st.write("### **Top 5 Similar Papers**")
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(df['title'] + ' ' + df['keywords'].apply(lambda x: ' '.join(x)))
        idx = filtered_df[filtered_df['title'] == selected_paper].index[0]
        cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
        similar_indices = cosine_sim.argsort()[-6:-1][::-1]
        for i in similar_indices:
            similarity_percentage = int(cosine_sim[i] * 100)  # Convert to Percentage(%)
            if cosine_sim[i] > 0.66:
                color = "green"
            elif cosine_sim[i] > 0.33:
                color = "orange"
            elif cosine_sim[i] == 0:
                color = "light gray"
            else:
                color = "red"
            st.write(f"- ##### **{df['title'][i]}** " f'<span style="color:{color};font-weight:bold;">{similarity_percentage}%</span>', unsafe_allow_html=True)
            st.write(f"Author : **{', '.join(df['author_full_name'][i])}**")
            st.write(f"Abstract : {df['abstract'][i]}")
            keywords_text_s = ", ".join(df['keywords'][i])
            st.write(f"Keywords: {keywords_text_s}")
            subject_text_s = ", ".join(df['subjects'][i])
            st.write(f"Subjects : {subject_text_s}")
            st.markdown("<hr>", unsafe_allow_html=True)
    else:
        st.write("No papers found based on your search query.")

############################################################### STATISTICS ###############################################################

elif page == "Statistics":
    
    st.title("Statistics & Analysis")
    
    #----------------------------- Get all keywords and subjects from the dataframe ------------------------------#
    all_keywords = [keyword for sublist in df['keywords'].dropna() for keyword in sublist]
    all_subjects = [subject for sublist in df['subjects'].dropna() for subject in sublist]

    df_exploded = df['author_full_name'].explode('author_full_name')
    
    total_papers = len(df)
    total_authors = len(df_exploded.unique())
    total_subjects = len(set(all_subjects))
    total_keywords = len(set(all_keywords))

    #----------------------------- Display Statistics ------------------------------#
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Total Papers", value=f"{total_papers}", delta=None)
    with col2:
        st.metric(label="Total Authors", value=f"{total_authors}", delta=None)
    with col3:
        st.metric(label="Total Subjects", value=f"{total_subjects}", delta=None)
    with col4:
        st.metric(label="Total Keywords", value=f"{total_keywords}", delta=None)
    st.dataframe(df)
    st.markdown("<hr>", unsafe_allow_html=True)

    ################################### Authors #######################################
    st.write("## Authors")

    #----------------------------- Top & Second Authors ------------------------------#
    top_authors = df_exploded.value_counts().head(1).reset_index()
    top_authors.columns = ['Author', 'Count']
    top_authors_display = f"{top_authors['Author'].iloc[0]} ({top_authors['Count'].iloc[0]})"

    second_authors = df_exploded.value_counts().iloc[1:2].reset_index()
    second_authors.columns = ['Author', 'Count']
    second_authors_display = f"{second_authors['Author'].iloc[0]} ({second_authors['Count'].iloc[0]})"
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Top Author", value=top_authors_display, delta=None)
    with col2:
        st.metric(label="Second Author", value=second_authors_display, delta=None)

    #--------------- Bar Chart : Top 30 Authors by Number of Papers ---------------#
    author_counts = df_exploded.value_counts().reset_index()
    author_counts.columns = ['Author', 'Paper Count']
    author_counts = author_counts.head(30)
    
    fig1 = px.bar(
        author_counts,
        x='Author',
        y='Paper Count',
        title="Top 30 Authors by Number of Papers",
        color='Paper Count',
        color_continuous_scale='Viridis',
        labels={'Paper Count': 'Number of Papers', 'Author': 'Author Name'},
        height=600,
        width=1800,
    )
    fig1.update_layout(
        xaxis_title="Authors",
        yaxis_title="Paper Count",
        xaxis_tickangle=-45,
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Arial", size=12),
        xaxis=dict(
            tickmode='linear', 
            tickangle=-45,
            automargin=True
        ),
        margin=dict(t=50, b=200)
    )
    st.plotly_chart(fig1)
    st.dataframe(author_counts)
    #--------------- Pie Chart : Subjects by Most Prolific Author ---------------#
    most_prolific_author = top_authors['Author'].iloc[0]
    prolific_author_subjects = df[df['author_full_name'].apply(lambda x: most_prolific_author in x)]['subjects']
    prolific_author_subjects = [subject for sublist in prolific_author_subjects.dropna() for subject in sublist]
    prolific_author_subject_freq = pd.Series(prolific_author_subjects).value_counts().reset_index()
    prolific_author_subject_freq.columns = ['Subject', 'Frequency']

    fig2 = px.pie(prolific_author_subject_freq, 
                  names='Subject', 
                  values='Frequency', 
                  title=f"Subjects by Most Prolific Author ({most_prolific_author})",
                  color_discrete_sequence=px.colors.sequential.Viridis)
    st.plotly_chart(fig2)

    #--------------- Bar Chart : Top 10 Subjects by Number of Authors ---------------#
    author_subject_paper_stats = df.explode('author_full_name').explode('subjects')
    author_subject_counts = author_subject_paper_stats.groupby(['author_full_name', 'subjects']).size().reset_index(name='paper_count')
    max_subject_per_author = author_subject_counts.loc[author_subject_counts.groupby('author_full_name')['paper_count'].idxmax()]
    subject_author_counts = max_subject_per_author.groupby('subjects').size().reset_index(name='author_count')
    top_10_subjects = subject_author_counts.nlargest(10, 'author_count').sort_values(by='author_count', ascending=True)

    fig3 = px.bar(
        top_10_subjects,
        x='author_count',
        y='subjects',
        orientation='h',
        color='author_count',
        color_continuous_scale='Viridis',
        labels={'author_count': 'Number of Authors', 'subjects': 'Subject'},
        title='Top 10 Subjects by Number of Authors'
    )
    st.plotly_chart(fig3)

    #--------------- Bar Chart : Top 10 Authors by Number of Papers in Top Subject ---------------#
    author_subject_paper_stats = df.explode('author_full_name').explode('subjects')
    subject_author_counts = author_subject_paper_stats.groupby('subjects')['author_full_name'].nunique().reset_index(name='author_count')
    top_subject = subject_author_counts.nlargest(1, 'author_count').iloc[0]['subjects']
    top_subject_authors = author_subject_paper_stats[author_subject_paper_stats['subjects'] == top_subject]
    author_paper_counts_in_top_subject = top_subject_authors.groupby('author_full_name').size().reset_index(name='paper_count')
    top_10_authors = author_paper_counts_in_top_subject.nlargest(10, 'paper_count')

    fig4 = px.bar(
        top_10_authors,
        x='author_full_name',
        y='paper_count',
        title=f"Top 10 Authors by Number of Papers in {top_subject}",
        color='paper_count',
        color_continuous_scale='Viridis',
        labels={'paper_count': 'Number of Papers', 'author_full_name': 'Author'},
        height=600,
        width=1500
    )
    fig4.update_layout(
        xaxis_title="Authors",
        yaxis_title="Paper Count",
        xaxis_tickangle=-45,
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Arial", size=12),
        margin=dict(t=50, b=200)
    )
    st.plotly_chart(fig4)

    #--------------- Metrics ---------------#
    author_subject_paper_stats = df.explode('author_full_name').explode('subjects')
    author_stats = author_subject_paper_stats.groupby('author_full_name').agg(
        average_subjects=('subjects', 'nunique'),
        paper_count=('title', 'count')
    ).reset_index()
    average_subjects = author_stats['average_subjects'].mean()
    average_papers = author_stats['paper_count'].mean()
    average_papers_top_subject = top_subject_authors.groupby('author_full_name').size().mean()
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Average Papers per Author (All)", value=f"{average_papers:.2f}")
    with col2:
        st.metric(label=f"In {top_subject} (Papers/Author)", value=f"{average_papers_top_subject:.2f}")

    #--------------- Average Subjects per Author ---------------#
    st.metric(label="Average Subjects per Author", value=f"{average_subjects:.2f}")

    #--------------- Scatter Plot : Relationship between Number of Papers and Number of Subjects by Author ---------------#
    fig5 = px.scatter(author_stats, 
                     x='paper_count', 
                     y='average_subjects', 
                     hover_name='author_full_name', 
                     title='Relationship between Number of Papers and Number of Subjects by Author',
                     labels={'paper_count': 'Number of Papers', 'average_subjects': 'Number of Subjects'},
                     color='average_subjects', 
                     color_continuous_scale='Viridis')
    st.plotly_chart(fig5)

    #--------------- Correlation Matrix ---------------#
    correlation_matrix = author_stats[['paper_count', 'average_subjects']].corr()

    fig6 = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values, 
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        colorscale='Viridis',
        colorbar=dict(title="Correlation Coefficient"),
        zmin=-1, zmax=1
    ))

    fig6.update_layout(
        title='Correlation Matrix: Paper Count vs Average Subjects',
        xaxis_title="Variables",
        yaxis_title="Variables",
        xaxis=dict(tickmode='array', tickvals=[0, 1], ticktext=correlation_matrix.columns),
        yaxis=dict(tickmode='array', tickvals=[0, 1], ticktext=correlation_matrix.columns),
        autosize=False,
        width=600,
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
    )
    st.plotly_chart(fig6)

    st.markdown("<hr>", unsafe_allow_html=True)

    ################################### Subjects #######################################
    st.write("## Subjects")

    #--------------- Top & Second Subjects ---------------#
    top_subjects = pd.Series(all_subjects).value_counts().head(2).reset_index()
    top_subjects.columns = ['Subject', 'Count']
    top_subjects_display = f"{top_subjects['Subject'].iloc[0]} ({top_subjects['Count'].iloc[0]})"
    second_subjects_display = f"{top_subjects['Subject'].iloc[1]} ({top_subjects['Count'].iloc[1]})" if len(top_subjects) > 1 else "N/A"

    col3, col4 = st.columns(2)
    with col3:
        st.metric(label="Top Subjects", value=top_subjects_display, delta=None)
    with col4:
        st.metric(label="Second Subjects", value=second_subjects_display, delta=None)

    #--------------- Bubble Chart : Subject Frequency ---------------#
    subject_freq = pd.Series(all_subjects).value_counts().reset_index()
    subject_freq.columns = ['Subject', 'Paper Count']
    fig1 = px.scatter(subject_freq, x='Subject', y='Paper Count', size='Paper Count', 
                      color='Paper Count', hover_name='Subject', title="Subjects by Number of Papers",
                      size_max=60, color_continuous_scale='Viridis',
                      width=800, height=800)
    st.plotly_chart(fig1)

    #--------------- Table : All Subjects with Paper Count ---------------#
    subject_freq = pd.Series(all_subjects).value_counts().reset_index()
    subject_freq.columns = ['Subject', 'Paper Count']
    st.dataframe(subject_freq)

    st.markdown("<hr>", unsafe_allow_html=True)

    ################################### Keywords #######################################
    st.write("## Keywords")

    top_keywords = pd.Series(all_keywords).value_counts().head(2).reset_index()
    top_keywords.columns = ['Keyword', 'Count']
    top_keywords_display = f"{top_keywords['Keyword'].iloc[0]} ({top_keywords['Count'].iloc[0]})"
    second_keywords_display = f"{top_keywords['Keyword'].iloc[1]} ({top_keywords['Count'].iloc[1]})" if len(top_keywords) > 1 else "N/A"

    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Top Keywords", value=top_keywords_display, delta=None)
    with col2:
        st.metric(label="Second Keywords", value=second_keywords_display, delta=None)
        
    #--------------- Word Cloud : Keywords ---------------#
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(" ".join(all_keywords))
    plt.figure(figsize=(8, 4))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

    #--------------- Table : All Keywords with Paper Count ---------------#
    keyword_freq = pd.Series(all_keywords).value_counts().reset_index()
    keyword_freq.columns = ['Keyword', 'Paper Count']
    st.dataframe(keyword_freq)