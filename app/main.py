import streamlit as st
import os
from utils import get_similar_images


# set paths
user_data_path = "app/user_imgs"
user_image_name = "user_plant.jpg"
features_data_path = "data/features.pkl"
json_data_path = "app/user_imgs/user_json_data.json"


def remove_data(image_path):
    if os.path.exists(image_path):
        os.remove(image_path)


def remove_json_data(json_data_path):
    if os.path.exists(json_data_path):
        os.remove(json_data_path)


def main():

    st.title("Similar plants searching")
    uploaded_file = st.file_uploader("Upload image of plant")

    if uploaded_file is not None:
        file_path = os.path.join(user_data_path, user_image_name)

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success("File saved!")
        st.spinner("Expect the result!", show_time=False)

        image_path = os.path.join(user_data_path, user_image_name)

        if get_similar_images(
            image_path=image_path,
            features_data_path=features_data_path,
            sim_imgs_num=5,
            report_path=json_data_path,
        ):

            with open(json_data_path, "rb") as file:
                st.download_button(
                    label="Download result",
                    data=file,
                    file_name="similar_images.json",
                )

            remove_data(image_path)
            remove_json_data(json_data_path)


if __name__ == "__main__":
    main()
