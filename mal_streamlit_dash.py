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



# Title
st.title('MyAnimeList Seasonal Stats')
st.write('A Streamlit app for MAL statistics, by season.')




# Season Select Dropdown
season_list = df['season_year'].unique()
season_list = [season for season in season_list if season is not None]

season_select = st.selectbox(
    'Select a Season',
    season_list,
    index=1)

df_copy = df.copy()
df_selected = df_copy[df_copy['season_year'] == season_select]




# Top Shows Table
sort_by_select = st.selectbox(
    'Select a Ranking Method',
    ['score', 'members']
)


df_top_10 = df_selected.sort_values(by=sort_by_select, ascending=False)[:10][['title', sort_by_select]].reset_index(drop=True)
df_top_10.index = df_top_10.index + 1

st.subheader(f'Top Shows in {season_select} by {sort_by_select.capitalize()}')

st.table(df_top_10.style.format(precision=2))


df_num_one = df_top_10.iloc[0]

# ------------- create number one show window here




# Summary Statistics: Total shows, avg score, std dev
# show_count, avg_score, std_dev_score = st.columns(3)

# with show_count:
#     st.metric(
#         'Number of Shows',
#         len(df_selected)
#         )

# with avg_score:
#     st.metric(
#         'Average Score',
#         round(df_selected['score'].mean(), 2)
#         )

# with std_dev_score:
#     st.metric(
#         'Std. Dev.',
#         round(df_selected['score'].std(), 2)
#         )





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
    marker_color = '#006dd5',
    marker=dict(
        line=dict(
            color='black',  # Border color
            width=0.5          # Border width
        )
    ),
    hovertemplate='<b>Number of shows with a %{y}.0 rating:<b> <br> %{x}<br>' + 
                '<extra></extra>'
))

# Update layout
fig.update_layout(
    title=f'Score Distribution in {season_select}',
    xaxis_title='Count',
    yaxis_title='Score',
    yaxis=dict(
        dtick=1
    ),
    width=800,
    height=500
)

#st.plotly_chart(fig)




# Score Distribution and Summary Statistics
x, y = st.columns([0.8, 0.2])

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
    



# Pie Charts

# Source Distribution
fig = px.pie(
    df_selected, 
    names='source',
    title='Source Material'
)

fig.update_traces(
    textinfo='label+percent', 
)

st.plotly_chart(fig)


# Type Distribution
fig2 = px.pie(
    df_selected, 
    names='type',
    title='Types'
)

fig2.update_traces(
    textinfo='label+percent', 
)

st.plotly_chart(fig2)



# Genre Statistics:

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
df_genre_stats.columns = ['count', 'average score']
df_genre_stats['percentage of shows (%)'] = df_genre_stats['count'] / len(df_selected) * 100
df_genre_stats = df_genre_stats.sort_values('count', ascending=False)
#df_genre_stats['proportion of shows'] = round(df_genre_stats['proportion of shows'], 2).astype(str) + '%'
df_genre_stats = df_genre_stats.style.format(precision=2)

st.subheader('Genres')

st.table(df_genre_stats)


