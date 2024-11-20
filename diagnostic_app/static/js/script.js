document.addEventListener("DOMContentLoaded", function() {
    const form = document.getElementById('dataForm');
    
    form.addEventListener('submit', function(event) {
        event.preventDefault();
        const formData = new FormData(form); 
        fetch('/process/', {
            method: 'POST',
            body: formData,
        })

        .then(response => response.json())  // Converte a resposta para JSON
        .then(data => {
            if (data.status === 'ok') {
                alert(data.message);  // Exibe a mensagem do backend (Aprovado ou Reprovado)
            } else {
                alert('Erro ao processar os dados');  // Caso ocorra um erro
            }
        })
        .catch(error => {
            alert('Erro de comunicação com o servidor');  // Exibe alerta de erro caso falhe a requisição
        });
        
    });
});
