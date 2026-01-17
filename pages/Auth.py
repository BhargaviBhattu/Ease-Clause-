import streamlit as st
import backend # Import the new backend module
import bcrypt # Needed for password hashing, used in app.py
from forgot_password import show_forgot_password_page # Import the function

# ---------------------------
# UI
# ---------------------------
# Initialize the database from the backend module
backend.init_db()

st.set_page_config(
    page_title="Login",
    layout="wide"
)

# Initialize a session state variable to control the view
if 'auth_view' not in st.session_state:
    st.session_state.auth_view = 'login'


# Render the page based on the current view state
if st.session_state.auth_view == 'login':
    if "user" in st.session_state and st.session_state.user:
        st.success(f"You are already logged in as {st.session_state.user['email']}")
        if st.button("Go to Main App"):
            st.switch_page("pages/Main_App.py")
    else:
        st.title("Sign In")

        with st.form("login_form"):
            # The input field is now correctly set to Email
            email = st.text_input("Email", placeholder="your@email.com")
            password = st.text_input("Password", type="password", placeholder="••••••••")
            submitted = st.form_submit_button("Sign In", type="primary")
            if submitted:
                # Get user data from the backend using the email
                data = backend.get_user(email)
                if data and bcrypt.checkpw(password.encode(), data[4]):
                    st.session_state.user = {"id": data[0], "email": data[3]}
                    st.success("Login successful. You can now access the app.")
                    st.switch_page("pages/Main_App.py")
                else:
                    st.error("Invalid email or password.")
                
        
        st.markdown("---")
        st.markdown("Forgot your password?")
        if st.button("Reset Password"):
            st.session_state.auth_view = 'forgot_password'
            # The new official way to rerun a Streamlit app.
            st.rerun()
        
        st.markdown("Don't have an account?")
        if st.button("Create an Account"):
            st.switch_page("pages/_Create_Account.py")
            
elif st.session_state.auth_view == 'forgot_password':
    show_forgot_password_page()
