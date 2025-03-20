import streamlit as st
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from PIL import Image
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import pandas as pd  
from email.mime.base import MIMEBase
from email import encoders

# Set page configuration
st.set_page_config(
    page_title="CanTrack",
    page_icon="🧠",
    initial_sidebar_state="expanded",
)

# Load the trained model
model = load_model("brain_tumor_cnn_model.h5")

# Function to preprocess the image
def preprocess_image(image):
    image = np.array(image)
    image = cv2.resize(image, (128, 128))
    image = image / 255.0  
    image = np.expand_dims(image, axis=0)  
    return image

# Function to send an email with optional attachments
def send_email(to_email, subject, body, attachments=[]):
    from_email = "vishwapadalia2004@gmail.com"  
    password = "agbd vqgb lony dqlt"

    msg = MIMEMultipart()
    msg["From"] = from_email
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    # Attach files if any
    for file in attachments:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(file.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f"attachment; filename={file.name}")
        msg.attach(part)

    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    server.login(from_email, password)
    server.sendmail(from_email, to_email, msg.as_string())
    server.quit()

# Streamlit UI
st.title("🧠 CanTrack")
st.write("🔍 Choose an option below to either detect a brain tumor, book an appointment, or view hospitals.")

# Sidebar for feature selection
option = st.radio("Select an option", ("🧠 Brain Tumor Detection", "📅 Book an Appointment", "🏥 Hospitals & Recognized Doctors"))

if option == "🧠 Brain Tumor Detection":
    st.subheader("🔬 Upload an MRI image to check for the presence of a brain tumor.")
    uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded MRI Image", use_column_width=True)
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        predicted_class = np.argmax(prediction)

        if predicted_class == 1:
            st.error("⚠️ Tumor detected! Please consult a healthcare provider immediately. 🏥")
        else:
            st.success("✅ No tumor detected. Keep up with regular health check-ups! 💪")

elif option == "📅 Book an Appointment":
    st.subheader("📞 Book an Appointment with a Specialist")

    doctors = [
        {"name": "Dr. John Doe", "specialization": "Neurosurgeon", "contact": "+1 555-123-4567", "email": "vishwapadalia2022@gmail.com"},
        {"name": "Dr. Jane Smith", "specialization": "Neurologist", "contact": "+1 555-987-6543", "email": "vishwapadalia2022@gmail.com"},
        {"name": "Dr. Robert Brown", "specialization": "Radiologist", "contact": "+1 555-456-7890", "email": "vishwapadalia2022@gmail.com"}
    ]

    doctor_options = [doctor["name"] for doctor in doctors]
    selected_doctor = st.selectbox("👨‍⚕️ Select a doctor", doctor_options)
    time_slots = ["10:00 AM", "11:00 AM", "3:00 PM", "4:00 PM", "5:00 PM", "7:00 PM"]

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

        # Add an optional file uploader for MRI scans or reports
        uploaded_images = st.file_uploader("📤 Upload MRI images or reports (Optional)", type=["jpg", "jpeg", "png", "pdf"], accept_multiple_files=True)

        submit_button = st.form_submit_button("📤 Submit Appointment Request")

        if submit_button:
            doctor = next(doctor for doctor in doctors if doctor["name"] == selected_doctor)
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
            
            # Send email with attachments
            send_email(doctor_email, subject, body_to_doctor, uploaded_images)

            user_subject = "📅 Appointment Request Confirmation"
            user_body = f"""
            Dear {name},

            Your appointment request has been received. Below are the details of the doctor:

            👨‍⚕️ **Doctor's Information**:
            Doctor: {doctor_name}
            Specialization: {doctor_specialization}
            Contact: {doctor_contact}

            🗓️ **Your Appointment Details**:
            Date: {date}
            Time Slot: {selected_time_slot}
            Message: {message}

            The doctor will contact you soon to confirm the appointment.

            Thank you for using our service. 🙏
            """
            send_email(email, user_subject, user_body)

            st.success("✅ Appointment request sent! You will receive a confirmation email. 📧")

elif option == "🏥 Hospitals & Recognized Doctors":
    st.subheader("🏥 List of Hospitals and Recognized Doctors")
    
    # Load hospital data from CSV
    hospitals = pd.read_csv("hospitals.csv")

    for _, row in hospitals.iterrows():
        st.write(f"### 🏥 {row['Name']}")
        st.write(f"📍 Location: {row['Location']}")
        st.write(f"📞 Contact: {row['Contact']}")
        st.write(f"🔬 Speciality: {row['Speciality']}")
        st.write(f"💰 Free/Low-Cost Treatment: {row['Free_Treatment']}")
        st.write("---")
