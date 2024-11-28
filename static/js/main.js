// main.js
document.addEventListener('DOMContentLoaded', function() {
    var uploadForm = document.querySelector('.upload-section form');
    uploadForm.addEventListener('submit', function(event) {
        // Prevent the default form submission
        event.preventDefault();

        // You can add more client-side validation here if needed

        // Submit the form data via an XMLHttpRequest or fetch() here
        // Or simply allow the form to submit if everything looks good
        uploadForm.submit();
    });

    // More JavaScript to handle the facial recognition and event selection
});
