import streamlit as st
import backend # Import the new backend module
# ---------------------------
# UI
# ---------------------------
st.set_page_config(
    page_title="Create an Account",
    layout="wide"
)

st.title("Create an Account")

with st.form("register_form"):
    col1, col2 = st.columns(2)
    with col1:
        r_first_name = st.text_input("First Name")
    with col2:
        r_last_name = st.text_input("Last Name")

    # The input field is now correctly set to Email
    r_email = st.text_input("New Email")
    r_pwd = st.text_input("New Password", type="password")
    r_confirm_pwd = st.text_input("Confirm Password", type="password")

    if st.form_submit_button("Create Account", use_container_width=True, type="primary"):
        if r_pwd != r_confirm_pwd:
            st.error("Passwords do not match.")
        else:
            # The add_user function from the backend now accepts email as the first argument
            ok = backend.add_user(r_email, r_pwd, r_first_name, r_last_name)
            if ok:
                st.success("Registration successful. You can now log in.")
                st.switch_page("pages/Auth.py")
            else:
                st.error("Email already exists. Try another one.")
                
st.markdown("---")
st.markdown("Already have an account?")
if st.button("Go to Login"):
    st.switch_page("pages/Auth.py")
