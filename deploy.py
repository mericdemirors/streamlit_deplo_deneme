import streamlit as st
import pandas as pd
import requests, re
import pickle
import math 
SEED = int(math.sqrt(201401004 + 191401009))

st.write("""# Simple Movie Prediction App\nThis app predicts the movie revenue!""")
st.sidebar.header('Movie information')


if st.button('need help with url?', on_click=None):
    img_paths = ["avengers1.png", "avengers2.png", "andreas1.png", "andreas2.png", "i_am_mother1.png", "i_am_mother2.png"]
    import random
    img_no = 2*(random.randint(0, len(img_paths)/2-1))
    
    from PIL import Image
    st.image(Image.open(img_paths[img_no]), caption='andreas', width=750)
    st.image(Image.open(img_paths[img_no+1]), caption='andreas', width=750)
    if st.button('thanks, I learnd:)', on_click=None):
        pass        
else:
    pass

# Taking input ---------------------------------------------------------------------------------------------------
def user_input_features():
    input_rating_url = st.text_input("IMDb site Ratings page url")
    st.write("\n(all other input other than **Year** gets disabled when url is provided. You should input **Year** always.)")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    
    if input_rating_url is None or input_rating_url == "":
        input_Year= st.sidebar.selectbox('Year', range(1990, 2023))
        input_Rating = st.sidebar.slider('Rating', min_value=float(0.0), value=float(5.0), max_value=float(10.0), step=0.1)
        input_1 = st.sidebar.number_input('Number of votes as 1', value=3000, format="%i")
        input_2 = st.sidebar.number_input('Number of votes as 2', value=3000, format="%i")
        input_3 = st.sidebar.number_input('Number of votes as 3', value=3000, format="%i")
        input_4 = st.sidebar.number_input('Number of votes as 4', value=3000, format="%i")
        input_5 = st.sidebar.number_input('Number of votes as 5', value=3000, format="%i")
        input_6 = st.sidebar.number_input('Number of votes as 6', value=3000, format="%i")
        input_7 = st.sidebar.number_input('Number of votes as 7', value=3000, format="%i")
        input_8 = st.sidebar.number_input('Number of votes as 8', value=3000, format="%i")
        input_9 = st.sidebar.number_input('Number of votes as 9', value=3000, format="%i")
        input_10 = st.sidebar.number_input('Number of votes as 10', value=3000, format="%i")
        data = {
            'Year' : input_Year,
            'Rating' : input_Rating,
            'Votes' : input_1 + input_2 + input_3 + input_4 + input_5 + input_6 + input_7 + input_8 + input_9 + input_10,
            '1': input_1, '2': input_2, '3': input_3, '4': input_4, '5': input_5, '6': input_6, '7': input_7, '8': input_8, '9': input_9, '10': input_10}
        return pd.DataFrame(data, index=[0])

    else:
        try:
            request_text = requests.get(input_rating_url).text
            star_votes = re.findall("<div class=\"leftAligned\">(.*?)</div>", request_text)
            rating = re.findall("span class=\"ipl-rating-star__rating\">(.*?)</span>", request_text)
            stars = [int(vote.replace(",","")) for vote in star_votes[1:11]]

            input_Year= st.sidebar.selectbox('Year', range(1990, 2023))
            [input_10, input_9, input_8, input_7, input_6, input_5, input_4, input_3, input_2, input_1] = stars
            input_Rating = float(rating[0])

            data = {
                'Year' : input_Year,
                'Rating' : input_Rating,
                'Votes' : input_1 + input_2 + input_3 + input_4 + input_5 + input_6 + input_7 + input_8 + input_9 + input_10,
                '1': input_1, '2': input_2, '3': input_3, '4': input_4, '5': input_5, '6': input_6, '7': input_7, '8': input_8, '9': input_9, '10': input_10}

            return pd.DataFrame(data, index=[0])
        except:
            st.write("Pwease input infowmations by hand")
            st.write("🥺👉👈")
            raise Exception("*Sowwy, unable to get data fwom pwovided page uwl*😔")


# Taking input ---------------------------------------------------------------------------------------------------
# ############################################################################################################## #
# ############################################################################################################## #
# ############################################################################################################## #
# Scaling input ---------------------------------------------------------------------------------------------------
def scale_raw_input(raw_input):
    data = raw_input.copy()
    import numpy as np
    year_revenue_dict = {1990: 0.7658344741262136, 1991: 0.6158904723529411, 1992: 0.5810284048958334, 1993: 0.5947457455973276, 1994: 0.5941740310769231, 1995: 0.6217016949917985, 1996: 0.6502561518881119, 1997: 0.6338443119205298, 1998: 0.8677550960544218, 1999: 0.6733860998742138, 2000: 0.7771750025433526, 2001: 0.7466757578888888, 2002: 0.708450799753397, 2003: 0.7759085865470852, 2004: 0.8238424626760563, 2005: 0.782264222, 2006: 0.7498834795081967, 2007: 0.6502655863192183, 2008: 0.7055149379672131, 2009: 0.8754953023278688, 2010: 0.6809290582777777, 2011: 0.9059954949253732, 2012: 0.811775069737705, 2013: 0.8438092751023868, 2014: 0.8119720837606839, 2015: 0.8807708197784809, 2016: 0.82369238359375, 2017: 1.0752321504301077, 2018: 0.8333133838255032, 2019: 0.9456890671153846, 2020: 0.3045369520779221, 2021: 0.9344049581962025, 2022: 0.9695750887288135}
    data['Year'] = data['Year'].map(year_revenue_dict)
    data["Rating"] = data["Rating"]/10
    data[['Votes', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']] = np.log2(data[['Votes', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']])

    min_max_scaler = pickle.load(open('MinMaxScaler.pickle', 'rb'))
    data[['Votes', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']]= min_max_scaler.transform(data[['Votes', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']])

    return data
# Scaling input ---------------------------------------------------------------------------------------------------
# ############################################################################################################## #
# ############################################################################################################## #
# ############################################################################################################## #
# Scaling input for NNs ------------------------------------------------------------------------------------------
def scale_raw_input_for_NN(raw_input):
    data = raw_input.copy()
    import numpy as np
    year_revenue_dict = {1990: 0.7658344741262136, 1991: 0.6158904723529411, 1992: 0.5810284048958334, 1993: 0.5947457455973276, 1994: 0.5941740310769231, 1995: 0.6217016949917985, 1996: 0.6502561518881119, 1997: 0.6338443119205298, 1998: 0.8677550960544218, 1999: 0.6733860998742138, 2000: 0.7771750025433526, 2001: 0.7466757578888888, 2002: 0.708450799753397, 2003: 0.7759085865470852, 2004: 0.8238424626760563, 2005: 0.782264222, 2006: 0.7498834795081967, 2007: 0.6502655863192183, 2008: 0.7055149379672131, 2009: 0.8754953023278688, 2010: 0.6809290582777777, 2011: 0.9059954949253732, 2012: 0.811775069737705, 2013: 0.8438092751023868, 2014: 0.8119720837606839, 2015: 0.8807708197784809, 2016: 0.82369238359375, 2017: 1.0752321504301077, 2018: 0.8333133838255032, 2019: 0.9456890671153846, 2020: 0.3045369520779221, 2021: 0.9344049581962025, 2022: 0.9695750887288135}
    data['Year'] = data['Year'].map(year_revenue_dict)
    data["Rating"] = data["Rating"]/10

    data["1%"] = data["1"] / data["Votes"]
    data["2%"] = data["2"] / data["Votes"]
    data["3%"] = data["3"] / data["Votes"]
    data["4%"] = data["4"] / data["Votes"]
    data["5%"] = data["5"] / data["Votes"]
    data["6%"] = data["6"] / data["Votes"]
    data["7%"] = data["7"] / data["Votes"]
    data["8%"] = data["8"] / data["Votes"]
    data["9%"] = data["9"] / data["Votes"]
    data["10%"] = data["10"] / data["Votes"]

    data[['Votes', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']] = np.log2(data[['Votes', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']])

    min_max_scaler = pickle.load(open('MinMaxScaler.pickle', 'rb'))
    data[['Votes', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']]= min_max_scaler.transform(data[['Votes', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']])

    return data
# Scaling input for NNs ------------------------------------------------------------------------------------------
# ############################################################################################################## #
# ############################################################################################################## #
# ############################################################################################################## #
# Setting data ---------------------------------------------------------------------------------------------------
raw_input = user_input_features()

st.subheader('Inputted Movie informations')
st.write(raw_input)

scaled_input = scale_raw_input(raw_input)
scaled_input_for_NN = scale_raw_input_for_NN(raw_input)
# Setting data ---------------------------------------------------------------------------------------------------
# ############################################################################################################## #
# ############################################################################################################## #
# ############################################################################################################## #
# Models  ---------------------------------------------------------------------------------------------------------
try:
    LR = pickle.load(open('LR.pickle', 'rb')) # good at middle part
except:
    pass

try:
    SVR_grid = pickle.load(open('SVR_grid.pickle', 'rb'))
    SVR = SVR_grid.best_estimator_ # good around high part
except:
    pass

try:
    RFR_grid = pickle.load(open('best_RFR.pickle', 'rb'))
    RFR = RFR_grid.best_estimator_ # good around high part
except:
    pass

try:
    import warnings
    warnings.filterwarnings("ignore")
    import tensorflow
    from tensorflow import keras;
    from tensorflow.keras.models import Sequential;
    from tensorflow.keras.layers import Dense, Dropout;
    from tensorflow.keras import Input

    model14 = Sequential([Input(shape=(23,))])
    model14.add(Dense(32, activation="tanh"))
    model14.add(Dropout(0.2, seed=SEED))
    model14.add(Dense(64, activation="tanh"))
    model14.add(Dropout(0.2, seed=SEED))
    model14.add(Dense(64, activation="tanh"))
    model14.add(Dense(32, activation="tanh"))
    model14.add(Dropout(0.2, seed=SEED))
    model14.add(Dense(13, activation="tanh"))
    model14.add(Dense(1))
    model14.compile(optimizer="Adam", loss=["mse", "mae"], metrics=["mae", "mse"])
    
    my_callbacks = [tensorflow.keras.callbacks.ModelCheckpoint( filepath = "kerasNN14.h5", monitor = "val_loss", verbose=0, save_best_only = True, save_weights_only = False, mode = "auto", save_freq = "epoch")]
    model14.load_weights('kerasNN14.h5')
except:
    pass

try:
    import warnings
    warnings.filterwarnings("ignore")
    import tensorflow
    from tensorflow import keras;
    from tensorflow.keras.models import Sequential;
    from tensorflow.keras.layers import Dense, Dropout;
    from tensorflow.keras import Input

    model16 = Sequential([Input(shape=(23,))])
    model16.add(Dense(32, activation="tanh"))
    model16.add(Dropout(0.2, seed=SEED))
    model16.add(Dense(64, activation="tanh"))
    model16.add(Dropout(0.2, seed=SEED))
    model16.add(Dense(64, activation="tanh"))
    model16.add(Dense(32, activation="tanh"))
    model16.add(Dropout(0.2, seed=SEED))
    model16.add(Dense(13, activation="tanh"))
    model16.add(Dense(1))
    model16.compile(optimizer="Adam", loss=["mse", "mae"], metrics=["mae", "mse"])

    my_callbacks = [tensorflow.keras.callbacks.ModelCheckpoint( filepath = "kerasNN16.h5", monitor = "val_loss", verbose=0, save_best_only = True, save_weights_only = False, mode = "auto", save_freq = "epoch")]
    model16.load_weights('kerasNN16.h5')
except:
    pass

try:
    import warnings
    warnings.filterwarnings("ignore")
    import tensorflow
    from tensorflow import keras;
    from tensorflow.keras.models import Sequential;
    from tensorflow.keras.layers import Dense, Dropout;
    from tensorflow.keras import Input

    model19 = Sequential([Input(shape=(23,))])
    model19.add(Dense(32, activation="sigmoid"))
    model19.add(Dropout(0.2, seed=SEED))
    model19.add(Dense(64, activation="sigmoid"))
    model19.add(Dense(64, activation="sigmoid"))
    model19.add(Dropout(0.2, seed=SEED))
    model19.add(Dense(64, activation="sigmoid"))
    model19.add(Dense(32, activation="sigmoid"))
    model19.add(Dropout(0.2, seed=SEED))
    model19.add(Dense(13, activation="sigmoid"))
    model19.add(Dense(1))
    model19.compile(optimizer="Adam", loss=["mse", "mae"], metrics=["mae", "mse"])

    my_callbacks = [tensorflow.keras.callbacks.ModelCheckpoint( filepath = "kerasNN19.h5", monitor = "val_loss", verbose=0, save_best_only = True, save_weights_only = False, mode = "auto", save_freq = "epoch")]
    model19.load_weights('kerasNN19.h5')
except:
    pass

try:
    import warnings
    warnings.filterwarnings("ignore")
    import tensorflow
    from tensorflow import keras;
    from tensorflow.keras.models import Sequential;
    from tensorflow.keras.layers import Dense, Dropout;
    from tensorflow.keras import Input

    model20 = Sequential([Input(shape=(23,))])
    model20.add(Dense(32, activation="sigmoid"))
    model20.add(Dropout(0.2, seed=SEED))
    model20.add(Dense(64, activation="sigmoid"))
    model20.add(Dense(64, activation="sigmoid"))
    model20.add(Dropout(0.2, seed=SEED))
    model20.add(Dense(64, activation="sigmoid"))
    model20.add(Dense(32, activation="sigmoid"))
    model20.add(Dropout(0.2, seed=SEED))
    model20.add(Dense(13, activation="sigmoid"))
    model20.add(Dense(1))
    model20.compile(optimizer="Adam", loss=["mse", "mae"], metrics=["mae", "mse"])

    my_callbacks = [tensorflow.keras.callbacks.ModelCheckpoint( filepath = "kerasNN20.h5", monitor = "val_loss", verbose=0, save_best_only = True, save_weights_only = False, mode = "auto", save_freq = "epoch"), tensorflow.keras.callbacks.ReduceLROnPlateau( monitor="val_loss", factor=0.3, patience=7, verbose=0, mode="auto", min_delta=0.0001, cooldown=5, min_lr=0)]
    model20.load_weights('kerasNN20.h5')
except:
    pass

try:
    import warnings
    warnings.filterwarnings("ignore")
    import tensorflow
    from tensorflow import keras;
    from tensorflow.keras.models import Sequential;
    from tensorflow.keras.layers import Dense, Dropout;
    from tensorflow.keras import Input

    model23 = Sequential([Input(shape=(23,))])
    model23.add(Dense(32, activation="sigmoid"))
    model23.add(Dropout(0.2, seed=SEED))
    model23.add(Dense(64, activation="sigmoid"))
    model23.add(Dense(32, activation="sigmoid"))
    model23.add(Dropout(0.2, seed=SEED))
    model23.add(Dense(13, activation="sigmoid"))
    model23.add(Dense(1))
    model23.compile(optimizer="Adam", loss=["mse", "mae"], metrics=["mae", "mse"])

    my_callbacks = [tensorflow.keras.callbacks.ModelCheckpoint( filepath = "kerasNN23.h5", monitor = "val_loss", verbose=1, save_best_only = True, save_weights_only = False, mode = "auto", save_freq = "epoch"), tensorflow.keras.callbacks.ReduceLROnPlateau( monitor="val_loss", factor=0.3, patience=7, verbose=1, mode="auto", min_delta=0.0001, cooldown=5, min_lr=0)]
    model23.load_weights('kerasNN23.h5')
except:
    pass
# Models  ---------------------------------------------------------------------------------------------------------

#st.subheader("scaled inputs:")
#st.write(scaled_input)
#st.subheader("scaled inputs for NN:")
#st.write(scaled_input_for_NN)


# Predicting  ---------------------------------------------------------------------------------------------------
LR_pred = LR.predict(scaled_input)
SVR_pred = SVR.predict(scaled_input)
#RFR_pred = RFR.predict(scaled_input)
model14_pred = model14.predict(scaled_input_for_NN)
model16_pred = model16.predict(scaled_input_for_NN)
model19_pred = model19.predict(scaled_input_for_NN)
model20_pred = model20.predict(scaled_input_for_NN)
model23_pred = model23.predict(scaled_input_for_NN)

#avg_pred = (LR_pred + SVR_pred + RFR_pred + model14_pred + model16_pred + model19_pred + model20_pred + model23_pred)/8
avg_pred = (LR_pred + SVR_pred + model14_pred + model16_pred + model19_pred + model20_pred + model23_pred)/7

revenue_pred = 2**(avg_pred[0][0])
desired_representation = "{:,.2f}".format(revenue_pred)

st.subheader('Prediction')
st.write("$", desired_representation)
# Predicting  ---------------------------------------------------------------------------------------------------
