import streamlit as st
import pickle
import pandas as pd
import requests
from nltk.stem.porter import PorterStemmer


TMDB_API_KEY = "b3d1c09882a6c002548499570898c89d"

def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&language=en-US"
    
    try:
        response = requests.get(url)
        response.raise_for_status() 
        data = response.json()
        
        poster_path = data.get('poster_path')
        overview = data.get('overview', 'Synopsis not available.') 
        
        full_path = "https://image.tmdb.org/t/p/w500/" + poster_path if poster_path else None
        
        return full_path, overview
    
    except requests.exceptions.RequestException:
        return None, "API connection error."
    except Exception:
        return None, "Data fetching error."




try:
    movies_dict = pickle.load(open('movie_dict.pkl', 'rb'))
    movies = pd.DataFrame(movies_dict)
    
    similarity = pickle.load(open('similarity.pkl', 'rb'))
    
except FileNotFoundError:
    st.error("Error: 'movie_dict.pkl' or 'similarity.pkl' files not found.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")
    st.stop()




def recommend(movie):
    """
    Takes a movie title and returns 5 most similar movies (titles and IDs).
    """
    recommended_movies = []
    recommended_movies_ids = []
    
    try:
        movie_index = movies[movies['title'] == movie].index[0]
    except IndexError:
        return recommended_movies, recommended_movies_ids

    distances = similarity[movie_index]
    sorted_movies = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    for i in sorted_movies:
        movie_data = movies.iloc[i[0]]
        recommended_movies.append(movie_data.title)
        recommended_movies_ids.append(movie_data.movie_id)

    return recommended_movies, recommended_movies_ids



st.set_page_config(layout="wide")
st.title('🎬 Movie recommendation system')

selected_movie_name = st.selectbox(
    'Select a movie you love:',
    movies['title'].values 
)

if st.button('Find Similar Movies', type="primary"):
    with st.spinner('Searching for the 5 most similar movies...'):
        
        recommendations, movie_ids = recommend(selected_movie_name)
        
        if recommendations:
            st.subheader(f'🎥 You might also like:')
            
            col1, col2, col3, col4, col5 = st.columns(5)
            cols = [col1, col2, col3, col4, col5]

            for i, (name, id) in enumerate(zip(recommendations, movie_ids)):
                with cols[i]:
                    poster_url, overview = fetch_poster(id) 
                    
                    st.caption(f"**{i+1}.** **{name}**")
                    
                    if poster_url:
                        st.image(poster_url)
                    else:
                        st.write("Poster not found 😔")


                    with st.expander("Read synopsis"):
                        st.markdown(f"*{overview}*")
