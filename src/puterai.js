import { puter } from "https://js.puter.com/v2/";


    // src/puterai.js
export const getPuteraiResponse = async (prompt) => {
    const response = await puter.ai.chat(prompt, { model: "gpt-4.1-nano" });
    return response; // Mengembalikan respons
};