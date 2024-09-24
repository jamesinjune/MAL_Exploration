import numpy as np
import pandas as pd
import plotly.graph_objs as go
import streamlit as st
# Page Configuration
st.set_page_config(layout='wide', page_title='SeasonalMAL')


# Read in data
file_url = 'https://raw.githubusercontent.com/jamesinjune/MAL_Exploration/refs/heads/main/df_mal_cleaned.csv'

df = pd.read_csv(file_url)


# Constants
graph_color_scheme = "#5193e9"

genre_list = ['Action',
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
              'Video Game', 'Visual Arts', 'Workplace']

season_list = df['season_year'].unique()
season_list = [season for season in season_list if season is not None]


# Title
st.title('SeasonalMAL')
st.markdown('A Streamlit app for MAL statistics, by season.')


# Widgets
season_select = st.selectbox(
    'Select a Season',
    season_list,
    index=1)

sort_by_select = st.selectbox(
    'Select a Ranking Method',
    ['score', 'members']
)


# Functions
def create_top_shows_table(df):
    def to_html_href(row):
        href_format = f'<a href="{row["url"]}" target="_blank" style="color: #236cdc; text-decoration: none;">{row["title"]}</a>'
        return href_format

    def capitalize_str(column_name):
        return column_name.capitalize()

    df_top_shows = df.copy()
    df_top_shows['title'] = df_top_shows.apply(to_html_href, axis=1)
    df_top_shows = df_top_shows.sort_values(by=sort_by_select, ascending=False)[
        :10][['title', sort_by_select]].reset_index(drop=True)
    df_top_shows.index = df_top_shows.index + 1
    top_shows_html = df_top_shows[['title', sort_by_select]].rename(
        columns=capitalize_str).to_html(escape=False, classes='top-shows-table')
    top_shows_html_styled = f"""
    <html>
    <head>
        <style>
            .top-shows-table {{
                width: 100%;
                border-collapse: collapse;
            }}
            .top-shows-table th, .top-shows-table td {{
                padding: 5px;
            }}
            .top-shows-table th {{
                color: #83858c;
                font-weight: normal;
            }}
            .top-shows-table td:nth-child(2), th:nth-child(2) {{
                text-align: left;
            }}
            .top-shows-table td:nth-child(3), th:nth-child(3) {{
                text-align: right;
            }}
            .top-shows-table tr:hover {{
                background-color: #cedeff;
                transition: background-color 0.3s ease;
            }}
        </style>
    </head>
    <body>
        {top_shows_html}
    </body>
    </html>
    """
    return top_shows_html_styled


def generate_num_one_stats(df):
    df_num_one = df.sort_values(by=sort_by_select, ascending=False).iloc[0]

    def get_genres(df):
        df_genres = df[genre_list]
        genre_index = df_genres[df_genres == 1].index
        genres = ', '.join(genre_index)
        return genres

    def get_episodes(df):
        if pd.isna(df['episodes']):
            episodes = df['episodes']
        else:
            episodes = int(df['episodes'])
        return episodes

    stats = {'genres': get_genres(df_num_one),
             'episodes': get_episodes(df_num_one),
             'title': df_num_one['title'],
             'url': df_num_one['url'],
             'score': df_num_one['score'],
             'members': df_num_one['members'],
             'type': df_num_one['type'],
             'studios': df_num_one['studios'],
             'source': df_num_one['source'],
             'duration': df_num_one['duration'],
             'aired': df_num_one['aired'],
             'images': df_num_one['images']}
    return stats


def image_display(stats_dict):
    html_image_display = f"""
    <div style="border: 5px solid #cedeff; padding: 7px; display: inline-block;">
        <a href="{stats_dict['url']}" target="_blank">
            <img src="{stats_dict['images']}" style="max-width: 230px; max-height: 310px; width: 100%" />
        </a>
    </div>
    """
    return html_image_display


def plot_score_distribution_histogram(df):
    bin_edges = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    hist_values, _ = np.histogram(df['score'], bins=bin_edges)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=hist_values,
        y=[f'{bin_edges[i]} ' for i in range(len(bin_edges)-1)],
        orientation='h',
        marker_color=graph_color_scheme,
        hovertemplate='Score: %{y}.0<br>Count: %{x}<br>' +
                    '<extra></extra>'
    ))

    fig.update_layout(
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
    return fig


def plot_source_distribution_histogram(df):
    df_source = df['source'].value_counts(
        sort=True, ascending=True).reset_index()

    sources = df_source['source']
    counts = df_source['count']
    percents = round(df_source['count'] / len(df) * 100, 2)

    fig = go.Figure(data=go.Bar(
        x=counts,
        y=sources,
        orientation='h',
        text=counts,
        texttemplate='%{text}',
        textposition='auto',
        marker_color=graph_color_scheme,
        hovertext=[f"Source: {src}<br>Count: {count}<br>Percent: {percent}%" for src,
                   count, percent in zip(sources, counts, percents)],
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
            'text': 'Source Material Distribution',
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
        height=499
    )
    return fig


def plot_type_distribution_pie(df):
    labels = df['type'].unique()
    values = df['type'].value_counts()

    fig = go.Figure(
        data=[go.Pie(
            labels=labels,
            values=values,
            textinfo='label',
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percent: %{percent:,.2%}<extra></extra>'
        )]
    )

    fig.update_layout(
        title={
            'text': 'Type Distribution',
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
        height=480
    )
    return fig


def seasonal_genre_stats(df):
    genre_columns = genre_list.copy()
    genre_columns.insert(0, 'score')

    df_genres = df[genre_columns]
    genre_counts = df_genres.iloc[:, 1:].sum()

    genre_average_scores = pd.Series(index=df_genres.columns[1:], dtype=float)

    for genre in df_genres.columns[1:]:
        genre_average_scores[genre] = df_genres[df_genres[genre] == 1]['score'].mean()

    df_genre_stats = pd.concat([genre_counts, genre_average_scores], axis=1)
    df_genre_stats.columns = ['Count', 'Average Score']
    df_genre_stats['Percentage of Shows (%)'] = df_genre_stats['Count'] / len(df) * 100
    df_genre_stats = df_genre_stats.sort_values('Count', ascending=False)
    df_genre_stats = df_genre_stats.style.format(precision=2)
    return df_genre_stats


def main():
    # Filter by Season Selection
    df_selected = df[df['season_year'] == season_select]

    # Top 10 Shows and Top Show Columns
    top_shows_table = create_top_shows_table(df_selected)
    num_one_stats = generate_num_one_stats(df_selected)
    num_one_image = image_display(num_one_stats)

    col1, col2 = st.columns([0.5, 0.4], gap='large')

    with col1:
        st.subheader(
            f'Top Shows in {season_select} by {sort_by_select.capitalize()}')
        st.markdown(top_shows_table, unsafe_allow_html=True)

    with col2:
        st.subheader(
            f'Top Show of {season_select} by {sort_by_select.capitalize()}')
        with st.container(border=True):
            subcol1, subcol2 = st.columns([0.44, 0.5])
            with subcol1:
                st.markdown(num_one_image, unsafe_allow_html=True)
                st.markdown(f'#### {num_one_stats["title"]}')
            with subcol2:
                st.markdown(
                    f'''
                    ##### Score: {num_one_stats["score"]}  
                    ##### Members: {num_one_stats["members"]}  
                    - **Type**: {num_one_stats["type"]}  
                    - **Episodes**: {num_one_stats["episodes"]}  
                    - **Aired**: {num_one_stats["aired"]}  
                    - **Studios**: {num_one_stats["studios"]}  
                    - **Source**: {num_one_stats["source"]}  
                    - **Duration**: {num_one_stats["duration"]}  
                    - **Genres**: {num_one_stats["genres"]}
                    '''
                )

    # Score Distribution and Summary Statistics Columns
    score_distribution_fig = plot_score_distribution_histogram(df_selected)

    col1, col2 = st.columns([0.8, 0.1])

    with col1:
        st.plotly_chart(score_distribution_fig)

    with col2:
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

    # Source Distribution and Type Distribution Columns
    source_distribution_fig = plot_source_distribution_histogram(df_selected)
    type_distribution_fig = plot_type_distribution_pie(df_selected)

    col1, col2 = st.columns(2, gap='large', vertical_alignment='bottom')

    with col1:
        st.plotly_chart(source_distribution_fig)

    with col2:
        st.plotly_chart(type_distribution_fig)

    # Genre Statistics and Distribution
    genre_stats = seasonal_genre_stats(df_selected)
    
    st.subheader('Genres')
    st.table(genre_stats)


if __name__ == "__main__":
    main()
