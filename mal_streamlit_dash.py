import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import plotly.offline as pyo
import seaborn as sns
import statsmodels

from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(layout="centered")

# Read in data
df = pd.read_json('C:/Users/croww/Desktop/df_mal_capitalized.json')



graph_color_scheme = "#5193e9"


# Title
st.title('MyAnimeList Seasonal Stats')
st.markdown('A Streamlit app for MAL statistics, by season.')




# Season Select Dropdown
season_list = df['season_year'].unique()
season_list = [season for season in season_list if season is not None]

season_select = st.selectbox(
    'Select a Season',
    season_list,
    index=1)

df_copy = df.copy()
df_selected = df_copy[df_copy['season_year'] == season_select]


df_genres = df_selected[['score', 'Action',
       'Adult Cast', 'Adventure', 'Anthropomorphic', 'Avant Garde',
       'Award Winning', 'Boys Love', 'CGDCT', 'Childcare', 'Combat Sports',
       'Comedy', 'Crossdressing', 'Delinquents', 'Detective', 'Drama', 'Ecchi',
       'Educational', 'Erotica', 'Fantasy', 'Gag Humor', 'Girls Love', 'Gore',
       'Gourmet', 'Harem', 'Hentai', 'High Stakes Game', 'Historical',
       'Horror', 'Idols (Female)', 'Idols (Male)', 'Isekai', 'Iyashikei',
       'Josei', 'Kids', 'Love Polygon', 'Magical Sex Shift', 'Mahou Shoujo',
       'Martial Arts', 'Mecha', 'Medical', 'Military', 'Music', 'Mystery',
       'Mythology', 'Organized Crime', 'Otaku Culture', 'Parody',
       'Performing Arts', 'Pets', 'Psychological', 'Racing', 'Reincarnation',
       'Reverse Harem', 'Romance', 'Romantic Subtext', 'Samurai', 'School',
       'Sci-Fi', 'Seinen', 'Shoujo', 'Shounen', 'Showbiz', 'Slice of Life',
       'Space', 'Sports', 'Strategy Game', 'Super Power', 'Supernatural',
       'Survival', 'Suspense', 'Team Sports', 'Time Travel', 'Vampire',
       'Video Game', 'Visual Arts', 'Workplace']]

df_genre_count = df_genres.iloc[:, 1:].sum()

df_genre_scores = pd.Series(index=df_genres.columns[1:], dtype=float)
for genre in df_genres.columns[1:]:
    genre_scores = df_genres.loc[df_genres[genre] == 1, 'score']
    df_genre_scores[genre] = genre_scores.mean()

df_genre_stats = pd.concat([df_genre_count, df_genre_scores], axis=1)
df_genre_stats.columns = ['Count', 'Average Score']
df_genre_stats['Percentage of Shows (%)'] = df_genre_stats['Count'] / len(df_selected) * 100
df_genre_stats = df_genre_stats.sort_values('Count', ascending=False)
df_genre_stats = df_genre_stats.style.format(precision=2)



# Top Shows Table
sort_by_select = st.selectbox(
    'Select a Ranking Method',
    ['score', 'members']
)

# Helper fn to create html hrefs for titles
def to_html_href(row):
    href_format = f'<a href="{row["url"]}" target="_blank" style="color: #236cdc; text-decoration: none;">{row["title"]}</a>'
    return href_format

def capitalize_str(column_name):
    return column_name.capitalize()

df_top_10 = df_selected.copy()
df_top_10['title'] = df_selected.apply(to_html_href, axis=1)
df_top_10 = df_top_10.sort_values(by=sort_by_select, ascending=False)[:10][['title', sort_by_select]].reset_index(drop=True)
df_top_10.index = df_top_10.index + 1
df_top_10_html = df_top_10[['title', sort_by_select]].rename(columns=capitalize_str).to_html(escape=False, classes='top-10-table')

styled_html = f"""
<html>
<head>
    <style>
        .top-10-table {{
            width: 100%;
            border-collapse: collapse;
        }}
        .top-10-table th, .top-10-table td {{
            padding: 5px;
            
        }}
        .top-10-table th {{
            color: #83858c;
            font-weight: normal;
        }}
        .top-10-table td:nth-child(2), th:nth-child(2) {{
            text-align: left;
        }}
        .top-10-table td:nth-child(3), th:nth-child(3) {{
            text-align: right;
        }}
        .top-10-table tr:hover {{
            background-color: #cedeff; /* Light gray background on hover */
            transition: background-color 0.3s ease; /* Smooth transition */
        }}
    </style>
</head>
<body>
    {df_top_10_html}
</body>
</html>
"""


#st.subheader(f'Top Shows in {season_select} by {sort_by_select.capitalize()}')
#st.markdown(styled_html, unsafe_allow_html=True)

# Top Show Display
df_num_one = df_selected.sort_values(by=sort_by_select, ascending=False).iloc[0]
top_title = df_num_one['title']
top_url = df_num_one['url']
top_score = df_num_one['score']
top_members = df_num_one['members']
if pd.isna(df_num_one['episodes']):
    top_episodes = df_num_one['episodes']
else:
    top_episodes = int(df_num_one['episodes'])
top_type = df_num_one['type']
top_studios = df_num_one['studios']
top_source = df_num_one['source']
top_duration = df_num_one['duration']
#if pd.isna(df_num_one['rank']):
#    top_rank = df_num_one['rank']
#else:
#    top_rank = '#' + str(int(df_num_one['rank']))
#top_popularity = '#' + str(df_num_one['popularity'])
df_num_one_genres = df_num_one[['Action',
       'Adult Cast', 'Adventure', 'Anthropomorphic', 'Avant Garde',
       'Award Winning', 'Boys Love', 'CGDCT', 'Childcare', 'Combat Sports',
       'Comedy', 'Crossdressing', 'Delinquents', 'Detective', 'Drama', 'Ecchi',
       'Educational', 'Erotica', 'Fantasy', 'Gag Humor', 'Girls Love', 'Gore',
       'Gourmet', 'Harem', 'Hentai', 'High Stakes Game', 'Historical',
       'Horror', 'Idols (Female)', 'Idols (Male)', 'Isekai', 'Iyashikei',
       'Josei', 'Kids', 'Love Polygon', 'Magical Sex Shift', 'Mahou Shoujo',
       'Martial Arts', 'Mecha', 'Medical', 'Military', 'Music', 'Mystery',
       'Mythology', 'Organized Crime', 'Otaku Culture', 'Parody',
       'Performing Arts', 'Pets', 'Psychological', 'Racing', 'Reincarnation',
       'Reverse Harem', 'Romance', 'Romantic Subtext', 'Samurai', 'School',
       'Sci-Fi', 'Seinen', 'Shoujo', 'Shounen', 'Showbiz', 'Slice of Life',
       'Space', 'Sports', 'Strategy Game', 'Super Power', 'Supernatural',
       'Survival', 'Suspense', 'Team Sports', 'Time Travel', 'Vampire',
       'Video Game', 'Visual Arts', 'Workplace']]
top_genre_index = df_num_one_genres[df_num_one_genres == 1].index
top_genres = ', '.join(top_genre_index)

# ^------------- create number one show window here
top_image = df_num_one['images']

# HTML and CSS to add a border around the image
html_image = f"""
<div style="border: 5px solid #CEDEFF; padding: 7px; display: inline-block;">
    <a href="{top_url}" target="_blank">
        <img src="{top_image}" style="max-width: 230px; max-height: 310px; width: 100%" />
    </a>
</div>
"""


col1, col2 = st.columns([0.5, 0.4], gap='large')

with col1:
    st.subheader(f'Top Shows in {season_select} by {sort_by_select.capitalize()}')
    st.markdown(styled_html, unsafe_allow_html=True)

with col2:
    st.subheader(f'Top Show of {season_select} by {sort_by_select.capitalize()}')
    with st.container(border=True):
        subcol1, subcol2 = st.columns([0.44, 0.5])
        with subcol1:
            st.markdown(html_image, unsafe_allow_html=True)
            st.markdown(f'#### {top_title}')
        with subcol2:
            st.markdown(
                f'''
                ##### Score: {top_score}  
                ##### Members: {top_members}  
                - **Episodes**: {top_episodes}  
                - **Type**: {top_type}  
                - **Studios**: {top_studios}  
                - **Source**: {top_source}  
                - **Duration**: {top_duration}  
                - **Genres**: {top_genres}
                '''
            )


#st.markdown('---')

# Score Distribution Histogram
# Define custom bin edges
bin_edges = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Compute histogram values
hist_values, _ = np.histogram(df_selected['score'], bins=bin_edges)

# Create histogram with custom bin edges
fig = go.Figure()

# Add histogram trace
fig.add_trace(go.Bar(
    x=hist_values,
    y=[f'{bin_edges[i]} ' for i in range(len(bin_edges)-1)],
    orientation='h',
    marker_color=graph_color_scheme,
    #marker=dict(
    #    line=dict(
    #        color='grey',  # Border color
    #        width=0.5          # Border width
    #    )
    #),
    hovertemplate='Score: %{y}.0<br>Count: %{x}<br>' + 
                '<extra></extra>'
))

# Update layout
fig.update_layout(
    #title=f'Score Distribution in {season_select}',
    title={
        'text': f'Score Distribution in {season_select}',
        'font': {
            'size': 22
        },
        'x': 0,
        'xanchor': 'left',
        'y': 0.90,
        'yanchor': 'top'
    },
    xaxis_title='Count',
    yaxis_title='Score',
    xaxis=dict(
        zeroline=True
    ),
    yaxis=dict(
        dtick=1
    ),
    width=1200,
    height=500,
    shapes=[
        {
            'type': 'line',
            'x0': -0.3,
            'x1': 1.5,
            'y0': 1.06,
            'y1': 1.06,
            'xref': 'paper',
            'yref': 'paper',
            'line': {
                'color': '#c9cacd',
                'width': 0.5
            }
        },
        {
            'type': 'line',
            'x0': 0,
            'x1': 1.5,
            'y0': 0,
            'y1': 0,
            'xref': 'paper',
            'yref': 'paper',
            'line': {
                'color': '#a19a9d',
                'width': 0.3
            }
        }
    ]
)

# Score Distribution and Summary Statistics
x, y = st.columns([0.8, 0.1])

with x:
    st.plotly_chart(fig)

with y:
    st.markdown("<br>" * 4, unsafe_allow_html=True)
    st.metric(
        'Number of Shows',
        len(df_selected)
    )
    st.metric(
        'Average Score',
        round(df_selected['score'].mean(), 2)
    )
    st.metric(
        'Std. Dev.',
        round(df_selected['score'].std(), 2)
    )
    


# Source Distribution Bar Chart
df_source = df_selected['source'].value_counts(sort=True, ascending=True).reset_index()

sources = df_source['source']
counts = df_source['count']
percents = round(df_source['count'] / len(df_selected) * 100, 2)

fig = go.Figure(data=go.Bar(
    x=counts,
    y=sources,
    orientation='h',
    text=counts,
    texttemplate='%{text}',
    textposition='auto',
    marker_color=graph_color_scheme,
    hovertext=[f"Source: {src}<br>Count: {count}<br>Percent: {percent}%" for src, count, percent in zip(sources, counts, percents)],
    hoverinfo='text'
))

fig.update_traces(
    textfont_size=12,
    textangle=0,
    textposition="outside",
    cliponaxis=False
)

fig.update_layout(
    title={
        'text': 'Source Material',
        'font': {
            'size': 22
        },
        'x': 0,
        'xanchor': 'left',
        'y': 0.93,
        'yanchor': 'top'
    },
    shapes=[
        {
            'type': 'line',
            'x0': -0.3,
            'x1': 1.5,
            'y0': 1.11,
            'y1': 1.11,
            'xref': 'paper',
            'yref': 'paper',
            'line': {
                'color': '#c9cacd',
                'width': 0.5
            }
        }
    ],
    #width=700,
    height=499
)

#st.plotly_chart(fig)


# Type Distribution Pie Chart
labels = df_selected['type'].unique()
values = df_selected['type'].value_counts()

fig2 = go.Figure(
    data=[go.Pie(
        labels=labels,
        values=values,
        textinfo='label',  # Adjust as needed for hover text
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percent: %{percent:,.2%}<extra></extra>'
    )]
)

fig2.update_layout(
    title={
        'text': 'Types',
        'font': {
            'size': 22
        },
        'x': 0,
        'xanchor': 'left',
        'y': 0.966,
        'yanchor': 'top'
    },
    shapes=[
        {
            'type': 'line',
            'x0': 0,
            'x1': 1.5,
            'y0': 1.18,
            'y1': 1.18,
            'xref': 'paper',
            'yref': 'paper',
            'line': {
                'color': '#c9cacd',
                'width': 0.5
            }
        }
    ],
    #width=,
    height=480
)

#st.plotly_chart(fig2)


col1, col2 = st.columns(2, gap='large', vertical_alignment='bottom')

with col1:
    st.plotly_chart(fig)

with col2:
    st.plotly_chart(fig2)

# Genre Statistics:
st.subheader('Genres')

st.table(df_genre_stats)



# Layout