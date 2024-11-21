document.addEventListener("DOMContentLoaded", function() {
    const form = document.getElementById('dataForm');
    
    form.addEventListener('submit', function(event) {
        event.preventDefault();
        const formData = new FormData(form); 
        fetch('/process/', {
            method: 'POST',
            body: formData,
        })

        .then(response => response.json())
        .then(data => {
            if (data.status === 'ok') {
                alert(data.message); 
            } else {
                alert('Erro ao processar os dados');
            }
        })
        .catch(error => {
            alert('Erro de comunicação com o servidor');
        });
        
    });
});
