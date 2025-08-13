import OpenAI from "openai";


const openai = new OpenAI({
    apiKey: localStorage.getItem('apiKey'),
    dangerouslyAllowBrowser:true // Ganti dengan kunci API Anda
  });
  export const getResponse = async (prompt) => {
    try {
      const response = await openai.chat.completions.create({
     // model: "gpt-oss-20b:free", 
        // Ganti dengan model yang Anda inginkan
       model: "gpt-3.5-turbo", 
        messages: [{ role: "user", content: prompt }],
      });
      return response.choices[0].message.content; // Mengembalikan konten respons
    } catch (error) {
      console.error("Error fetching response from OpenAI:", error);
      throw error; // Melempar kembali error untuk ditangani di tempat lain
    }
  };
