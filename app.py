import streamlit as st
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from PIL import Image
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Load the trained model
model = load_model("brain_tumor_cnn_model.h5")

# Function to preprocess the image


def preprocess_image(image):
    image = np.array(image)  # Convert the image to a NumPy array
    image = cv2.resize(image, (128, 128))  # Resize to 128x128
    image = image / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to send an email


def send_email(to_email, subject, body):
    from_email = "vishwapadalia2004@gmail.com"  # Replace with your email address
    # Replace with your email password or app-specific password
    password = "agbd vqgb lony dqlt"

    # Setup email server
    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    server.login(from_email, password)

    # Prepare the email
    msg = MIMEMultipart()
    msg["From"] = from_email
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    # Send the email
    server.sendmail(from_email, to_email, msg.as_string())
    server.quit()


# Streamlit UI
st.title("🧠 Brain Tumor Detection & Appointment Booking App")
st.write("🔍 Choose an option below to either detect a brain tumor or book an appointment for consultation.")

# Sidebar for feature selection
option = st.radio("Select an option",
                  ("🧠 Brain Tumor Detection", "📅 Book an Appointment"))

if option == "🧠 Brain Tumor Detection":
    st.subheader(
        "🔬 Upload an MRI image to check for the presence of a brain tumor.")

    # File upload for image
    uploaded_file = st.file_uploader(
        "Choose an MRI image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded MRI Image", use_column_width=True)

        # Preprocess the image and make prediction
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        predicted_class = np.argmax(prediction)

        # Display the result with emojis for visual feedback
        if predicted_class == 1:
            st.error(
                "⚠️ Tumor detected! Please consult a healthcare provider immediately. 🏥")
        else:
            st.success(
                "✅ No tumor detected. Keep up with regular health check-ups to stay healthy! 💪")

elif option == "📅 Book an Appointment":
    st.subheader("📞 Book an Appointment with a Specialist")

    # Doctor information
    doctors = [
        {"name": "Dr. John Doe", "specialization": "Neurosurgeon",
            "contact": "+1 555-123-4567", "email": "krishnabhatt0701@gmail.com"},
        {"name": "Dr. Jane Smith", "specialization": "Neurologist",
            "contact": "+1 555-987-6543", "email": "krishnabhatt863@gmail.com"},
        {"name": "Dr. Robert Brown", "specialization": "Radiologist",
            "contact": "+1 555-456-7890", "email": "indiasprimestore2020@gmail.com"}
    ]

    # Display doctor information
    doctor_options = [doctor["name"] for doctor in doctors]
    selected_doctor = st.selectbox("👨‍⚕️ Select a doctor", doctor_options)

    # Define available time slots
    time_slots = ["10:00 AM", "11:00 AM",
                  "3:00 PM", "4:00 PM", "5:00 PM", "7:00 PM"]

    # Appointment booking form
    with st.form(key="appointment_form"):
        name = st.text_input("👤 Your Name")
        email = st.text_input("📧 Your Email Address")
        contact = st.text_input("📞 Your Contact Number")
        city = st.text_input("🏙️ City")
        state = st.text_input("🌆 State")
        country = st.text_input("🌍 Country")
        date = st.date_input("📅 Preferred Appointment Date")
        selected_time_slot = st.selectbox("⏰ Select a time slot", time_slots)
        message = st.text_area("💬 Message (optional)")

        # Submit button
        submit_button = st.form_submit_button("📤 Submit Appointment Request")

        if submit_button:
            # Find selected doctor's details
            doctor = next(
                doctor for doctor in doctors if doctor["name"] == selected_doctor)
            doctor_email = doctor["email"]
            doctor_name = doctor["name"]
            doctor_specialization = doctor["specialization"]
            doctor_contact = doctor["contact"]

            subject = "🩺 New Appointment Request"
            body_to_doctor = f"""
            Appointment Request from {name} ({email}):

            📌 **Patient Details**:
            Name: {name}
            Email: {email}
            Contact: {contact}
            City: {city}
            State: {state}
            Country: {country}

            🗓️ **Appointment Details**:
            Doctor: {selected_doctor}
            Specialization: {doctor_specialization}
            Date: {date}
            Time Slot: {selected_time_slot}
            Message: {message}

            Please reach out to confirm the appointment.
            """
            # Send email to doctor
            send_email(doctor_email, subject, body_to_doctor)

            # Confirmation email to user with doctor's details
            user_subject = "📅 Appointment Request Confirmation"
            user_body = f"""
            Dear {name},

            Your appointment request has been received. Below are the details of the doctor you selected:

            👨‍⚕️ **Doctor's Information**:
            Doctor: {doctor_name}
            Specialization: {doctor_specialization}
            Contact: {doctor_contact}

            🗓️ **Your Appointment Details**:
            Contact: {contact}
            City: {city}
            State: {state}
            Country: {country}
            Date: {date}
            Time Slot: {selected_time_slot}
            Message: {message}

            The doctor will contact you soon to confirm the appointment.

            Thank you for using our service. 🙏
            """
            send_email(email, user_subject, user_body)

            st.success(
                f"✅ Appointment request sent! You will receive a confirmation email shortly. 📧")
