document.addEventListener('DOMContentLoaded', function () {
  // For signup form
  const signupForm = document.getElementById("signupForm");
  if (signupForm) {
    signupForm.addEventListener("submit", function (event) {
      event.preventDefault();  // Prevent default form submission
      const email = document.getElementById("signupEmail").value;
      const password = document.getElementById("signupPassword").value;

      fetch('http://127.0.0.1:5000/signup', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          email: email,
          password: password
        })
      })
      .then(response => {
        if (!response.ok) {
          return response.text();  // Read response as text
        }
        return response.json();  // Parse JSON if OK
      })
      .then(data => {
        if (typeof data === "string") {
          console.error("Error response:", data);
          alert("Error: " + data);
        } else {
          console.log('Signup response:', data);
          if (data.message === 'Signup successful') {
            window.location.href = '/login'; // Redirect to login if successful
          } else {
            alert(data.message);  // Show error message from backend
          }
        }
      })
      .catch(error => {
        console.error('Error during signup:', error);
      });
    });
  }

  // For login form
  const loginForm = document.getElementById("loginForm");
  if (loginForm) {
    loginForm.addEventListener("submit", function (e) {
      e.preventDefault();

      const email = document.getElementById("loginEmail").value;
      const password = document.getElementById("loginPassword").value;

      fetch("http://127.0.0.1:5000/login", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ email, password })
      })
      .then(res => res.json())
      .then(data => {
        console.log("Login response:", data);
        if (data.message === "Login successful") {
          window.location.href = data.redirect_url;
        } else {
          alert(data.message);
        }
      })
      .catch(err => console.error("Login error:", err));
    });
  }
});
