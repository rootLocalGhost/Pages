/* @refresh reload */
import { render } from "solid-js/web";
import App from "./App";
import "./App.css";

const root = document.getElementById("root");

if (import.meta.env.DEV && !(root instanceof HTMLElement)) {
  throw new Error(
    "Root element not found. Check if index.html has div#root."
  );
}

render(() => <App />, root!);
