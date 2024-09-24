# SeasonalMAL
[SeasonalMAL](https://mal-seasonal-stats.streamlit.app/) is an interactive summary of shows and statistics on MyAnimeList, by season.

## Description
In Japan, broadcasting seasons are approximately three months long and correspond to the natural seasons of the year.

SeasonalMAL shows you the distribution of score, source material, media type, and genre for any given season on MyAnimeList, as well as the top shows of the season.

## Tools Used
- Python - [Data extraction/cleaning process](notebooks/MAL_API_Data_Extraction.ipynb)
- Python/Streamlit - [Application](mal_streamlit_dash.py)

Data was extracted from MyAnimeList using [Jikan API](https://jikan.moe/), an unofficial API for MyAnimeList. Documentation can be found [here](https://docs.api.jikan.moe/).

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for more information.
