document.addEventListener("DOMContentLoaded", function () {
    const ws = new WebSocket("ws://localhost:8000/ws");

    ws.onmessage = function(event) {
        const modelInfo = JSON.parse(event.data);
        displayModelInfo(modelInfo);
    };

    function displayModelInfo(modelInfo) {
        const container = document.getElementById("model-visualization");
        container.innerHTML = "<pre>" + JSON.stringify(modelInfo, null, 2) + "</pre>";
    }
});
