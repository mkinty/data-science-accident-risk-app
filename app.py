#############                                         #######################
                         #######   Librairies   ######
##############                                      ########################
import os
from pathlib import Path
from PIL import Image
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


#############                                         #######################
                         #######   main   ######
##############                                      ########################

def main():
    """News Classifier App with Streamlit"""



    activities = ["Accueil", "Datasets", "Data Viz", "Models", "Prediction", "Auteur"]
    choice = st.sidebar.selectbox("choose Activity", activities)

    all_city = ["Atlanta", "Austin", "Charlotte", "Dallas", "Houston", "LosAngeles"]
    all_data = ["traffic_data_20180601_20180609.csv", "poi_data_20180601_20180609.csv", "weather_data_20180601_20180609.csv", "TrafficWeatherEvent_June18_Aug18_Publish.csv"]
    all_attribut = ["City", "weekday", "hour", "weekday_hour" ]
    site_data  = ["traffic_data_20180601_20180609.csv", "weather_data_20180601_20180609.csv", "poi_data_20180601_20180609.csv"]


    tw_data = ["TrafficWeatherEvent_June18_Aug18_Publish.csv"]


    #############                                         #######################
                         #######   Datasets   ######
    ##############                                      ########################


    for dataset in tw_data:
        def load_twpoi_data():
            bikes_data_path = Path() /"data/clean_twpoi_data/{}".format(dataset)
            data = pd.read_csv(bikes_data_path)
            return data
        TrafficWeatherEvent = load_twpoi_data()

    if choice == "Accueil" :
        #st.title("Streamlit ML App")
        html_temp = """
        <div style = "background-color : magenta ; padding:10px;">
        <h2> PRÉVISION DU RISQUE D'ACCIDENT DE TRAFIC</h2>
        </div>
        """
        st.markdown(html_temp, unsafe_allow_html=True)

        image = Image.open('images/index.jpg')
        st.image(image, use_column_width = True)

    def file_selector(folder_path='./data/predict_data'):
        filenames = os.listdir(folder_path)
        selected_filename = st.selectbox("Selectionner un des fichier .csv ci-dessous pour tester faire la prédiction", filenames)
        return os.path.join(folder_path, selected_filename)


    if choice == "Datasets":
        st.subheader("Jeux de Données : https://smoosavi.org/datasets/lstw \n "
                     "https://github.com/mhsamavatian/DAP/tree/master/data")



        data_choice = st.selectbox("file name", site_data)

        if st.checkbox("Show Dataset"):
            def load_twpoi_data():
                bikes_data_path = Path() / "data/traffic_weather_poi_data/{}".format(data_choice)
                data = pd.read_csv(bikes_data_path)
                return data
            df = load_twpoi_data()
            st.write(df.tail(10))


    #############                                         #######################
                           #######   Data Viz   ######
    ##############                                      ########################

    for city in all_city:
        @st.cache
        def load_data():
            bikes_data_path = Path() / 'data/temporary_accident/TA_{}_20180601_20180609.csv'.format((city))
            data = pd.read_csv(bikes_data_path)
            return data

        if city == "Atlanta":
            Atlanta = load_data()
        elif city == "Austin":
            Austin = load_data()
        elif city == "Dallas":
            Dallas = load_data()
        elif city == "Charlotte":
            Charlotte = load_data()
        elif city == "LosAngeles":
            LosAngeles = load_data()
        elif city == "Houston":
            Houston = load_data()

    def load_traffic_accident_data():
        bikes_data_path = Path() / "data/traffic_accident_data/traffic_accident_data_20180601_20180609.csv"
        data = pd.read_csv(bikes_data_path)
        return data
    traffic_accident_data = load_traffic_accident_data()
    weekday_hour = traffic_accident_data.groupby("weekday hour".split()).apply(lambda x: len(x))
    corr_weekday_hour = weekday_hour.unstack()

    if choice == "Data Viz":
        st.info("Les Accidents de Trafic entre 01 Juin 2018 et le 31 Août 2018")
        type = ["Graphe", "Map", "TrafficWeatherEvent_June18_Aug18_Publish.csv"]
        incident_choice = st.selectbox("File name", type)


        #if st.button("visualiser"):
        if incident_choice == "Graphe":
            #st.write(traffic_accident_data.head(4))
            attribut_choice = st.selectbox("Attribute name", all_attribut)
            if attribut_choice in ["City", "weekday", "hour"]:
                fig1 = px.histogram(traffic_accident_data, x="{}".format(attribut_choice), y="Type", color="Type",
                                    marginal="rug", hover_data=traffic_accident_data.columns)
                barplot_chart = st.write(fig1)
            elif attribut_choice == "weekday_hour":
                #st.write(corr_weekday_hour)
                st.dataframe(corr_weekday_hour.style.highlight_max(axis=0))

        if incident_choice == "Map":
            attribut_choice = st.selectbox("Attribute name", all_city)
            if st.button("Visualiser"):
                if attribut_choice == "Atlanta":
                    Atlanta = Atlanta.loc[:, ["LocationLat", "LocationLng"]]
                    Atlanta.columns = ["lat", "lon"]
                    st.map(Atlanta)
                elif attribut_choice == "Austin":
                    Austin = Austin.loc[:, ["LocationLat", "LocationLng"]]
                    Austin.columns = ["lat", "lon"]
                    st.map(Austin)
                elif attribut_choice == "Dallas":
                    data = Dallas.loc[:, ["LocationLat", "LocationLng"]]
                    data.columns = ["lat", "lon"]
                    st.map(data)
                elif attribut_choice== "Charlotte":
                    data = Charlotte.loc[:, ["LocationLat", "LocationLng"]]
                    data.columns = ["lat", "lon"]
                    st.map(data)
                elif attribut_choice == "LosAngeles":
                    data = LosAngeles.loc[:, ["LocationLat", "LocationLng"]]
                    data.columns = ["lat", "lon"]
                    st.map(data)
                elif attribut_choice == "Houston":
                    data = Houston.loc[:, ["LocationLat", "LocationLng"]]
                    data.columns = ["lat", "lon"]
                    st.map(data)

        elif incident_choice == "TrafficWeatherEvent_June18_Aug18_Publish.csv":
            if st.checkbox("Show Dataset"):
                st.write(TrafficWeatherEvent.tail(10))
                if st.checkbox("Le taux de zero accident"):
                    st.write("Le taux de zero accident  est égal à :", float(TrafficWeatherEvent[TrafficWeatherEvent['predicted_accident']==0].shape[0])/TrafficWeatherEvent.shape[0])


    #############                                         #######################
                       #######   Models   ######
    ##############                                      ########################

    alg = ['Logistic Regression', 'Gradient Boosting Classifier']

    # Standardisize
    sc_x = StandardScaler()

    # # Gradient Boosting Classifier
    gb_clf = GradientBoostingClassifier(n_estimators=120, learning_rate=0.95, random_state=42)

    # Modele de regression Logistique
    classifier = LogisticRegression()




    if choice == "Models" :
        st.info("Machine learning")
        model = st.selectbox('Which algorithm?', alg)

        # Model de Regression logistic
        if model == 'Logistic Regression':
            city_choice = st.selectbox("file name", tw_data)
            if city_choice in tw_data:
                df = pd.read_csv("data/clean_twpoi_data/{}".format(city_choice))
                st.write(df.tail(5))
                if st.button("Classify"):
                    X = df.loc[:, df.columns != "predicted_accident"]
                    y = df.loc[:, df.columns == "predicted_accident"]

                    # Split in train/test
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)

                    X_train = sc_x.fit_transform(X_train)
                    X_test = sc_x.transform(X_test)
                    classifier.fit(X_train, y_train)

                    pred_classifier = classifier.predict(X_test)

                    # Accuracy score
                    st.write("Accuracy score (training) :", classifier.score(X_train, y_train))
                    st.write("Accuracy score (validation) ", classifier.score(X_test, y_test))

                    # Confusion matrix
                    cm_classifier = confusion_matrix(y_test, pred_classifier)
                    st.write('Confusion matrix: ', cm_classifier)


        # Grandient Boosting Classifier
        elif model == 'Gradient Boosting Classifier':
            city_choice = st.selectbox("file name", tw_data)
            if city_choice in tw_data:
                df = pd.read_csv("data/clean_twpoi_data/{}".format(city_choice))
                st.write(df.tail(5))
                if st.button("Classify"):
                    X = df.loc[:, df.columns != "predicted_accident"]
                    y = df.loc[:, df.columns == "predicted_accident"]

                    # Split in train/test
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)



                    X_train = sc_x.fit_transform(X_train)
                    X_test = sc_x.transform(X_test)


                    gb_clf.fit(X_train, y_train)
                    # Accuracy score
                    st.write("Accuracy score (training) :", gb_clf.score(X_train, y_train))
                    st.write("Accuracy score (validation) ", gb_clf.score(X_test, y_test))
                    pred_gb_clf = gb_clf.predict(X_test)

                    # Confusion matrix
                    cm_gb_clf = confusion_matrix(y_test, pred_gb_clf)
                    st.write('Confusion matrix: ', cm_gb_clf)

    #############                                         #######################
                        #######   Prediction   ######
    ##############                                      ########################



    if choice == "Prediction" :
        st.info("Prédire le risque d'accident de trafic")
        df = pd.read_csv("data/clean_twpoi_data/TrafficWeatherEvent_June18_Aug18_Publish.csv")
        #st.multiselect("Les variables du fichier", df.columns.tolist())
        filename = file_selector()
        st.write('You selected `%s`' % filename)

        if st.checkbox("Show Dataset"):
            X_valid = pd.read_csv("{}".format(filename))
            st.write(X_valid)

            model = st.selectbox('Which algorithm ?', alg)

            if model == 'Logistic Regression':
                X = df.loc[:, df.columns != "predicted_accident"]
                y = df.loc[:, df.columns == "predicted_accident"]

                # Split in train/test
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)

                X_train = sc_x.fit_transform(X_train)
                X_test = sc_x.transform(X_test)
                X_valid = sc_x.transform(X_valid)

                classifier.fit(X_train, y_train)

                pred_valid = classifier.predict(X_valid)
                pred_valid =(pred_valid == 1)
                if st.button("Evaluer"):
                    st.write("Le risque d'accident de trafic : \n {} ".format(pred_valid))

            elif model == 'Gradient Boosting Classifier':

                X = df.loc[:, df.columns != "predicted_accident"]
                y = df.loc[:, df.columns == "predicted_accident"]

                # Split in train/test
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)

                X_train = sc_x.fit_transform(X_train)
                X_test = sc_x.transform(X_test)
                X_valid = sc_x.transform(X_valid)

                gb_clf.fit(X_train, y_train)

                pred_valid = gb_clf.predict(X_valid)
                pred_valid =(pred_valid == 1)
                if st.button("Evaluer"):
                    st.write("Le risque d'accident de trafic : \n {} ".format(pred_valid))

    if choice == "Auteur" :
        st.info(" Moustapha KINTY \n" 
                "[`GitHub`](https://github.com/mkinty), [`LinkedIn`](https://www.linkedin.com/in/moustapha-kinty-8288b0153/), \n"   "E-mail : kintymoustapha@gmail.com")


        image = Image.open('images/mkinty.jpg')
        st.image(image, )

if __name__ == "__main__":
    main()
