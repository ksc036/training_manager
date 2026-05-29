import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./styles.css";

const container = document.getElementById("app-root");

if (!container) {
  throw new Error("Missing app root element with id 'app-root'.");
}

ReactDOM.createRoot(container).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);

