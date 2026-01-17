# pages/admin_dashboard.py
import streamlit as st
from backend import (
    get_all_users,
    # get_user_logs,
    get_user_documents,
    verify_user,
    is_admin,
    setup
)

# Initialize DB & admin
setup()

# ---------------------------------------------------------
# Admin Login Page
# ---------------------------------------------------------
def admin_login():
    st.title("👩‍💼 Admin Login - ClauseEase Dashboard")
    st.markdown("Please enter your admin credentials to access the dashboard.")

    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if verify_user(email, password) and is_admin(email):
            st.session_state["is_admin"] = True
            st.session_state["admin_email"] = email
            st.success("✅ Login successful!")
            st.stop()  # Stop current run; page reruns with updated session state
        else:
            st.error("❌ Invalid credentials or not an admin!")

# ---------------------------------------------------------
# Admin Dashboard Page
# ---------------------------------------------------------
def admin_dashboard():
    st.title("📊 Admin Dashboard - ClauseEase")
    st.sidebar.success(f"Logged in as: {st.session_state['admin_email']}")

    tab1, tab2, tab3 = st.tabs(["👥 Manage Users", "📜 View Logs", "📂 User Documents"])

    # --- Tab 1: Manage Users ---
    with tab1:
        st.subheader("Registered Users")
        users = get_all_users()
        if not users:
            st.info("No users found.")
        else:
            for u in users:
                st.markdown(f"""
                - **Email:** {u[3]}  
                - **Name:** {u[1]} {u[2]}  
                - **Admin:** {'✅' if u[4] == 1 else '❌'}
                ---""")

    # --- Tab 2: View Logs ---
    with tab2:
        st.subheader("View User Activity Logs")
        user_email_log = st.text_input("Enter user email to view logs", key="log_email")
        if st.button("Fetch Logs", key="fetch_logs"):
            logs = get_user_logs(user_email_log)
            if logs:
                for log in logs:
                    st.markdown(f"""
                    **Simplified Text:**  
                    `{log[2]}`  
                    **Level:** {log[0]}  
                    **Timestamp:** {log[3]}  
                    ---""")
            else:
                st.warning("No logs found for this user.")

    # --- Tab 3: User Documents ---
    with tab3:
        st.subheader("Fetch Documents Uploaded by Users")
        user_email_doc = st.text_input("Enter user email to fetch documents", key="doc_email")
        if st.button("Fetch Documents", key="fetch_docs"):
            docs = get_user_documents(user_email_doc)
            if docs:
                for doc in docs:
                    st.markdown(f"""
                    **Filename:** {doc[1]}  
                    **MIME Type:** {doc[2]}  
                    **Uploaded At:** {doc[4]}  
                    """)
                    st.download_button(
                        label="Download Document",
                        data=doc[3],
                        file_name=doc[1],
                        mime=doc[2]
                    )
            else:
                st.warning("No documents found for this user.")

    # --- Logout ---
    if st.button("🚪 Logout", key="logout_admin"):
        st.session_state.pop("is_admin", None)
        st.session_state.pop("admin_email", None)
        st.success("Logged out successfully!")
        st.stop()  # Stop current run to rerun the page without admin session

# ---------------------------------------------------------
# Main Logic
# ---------------------------------------------------------
def show_page():
    if "is_admin" in st.session_state and st.session_state["is_admin"]:
        admin_dashboard()
    else:
        admin_login()

# Run the page
if __name__ == "__main__":
    show_page()

