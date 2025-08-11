import OpenAI from "openai";
const openai = new OpenAI({
    apiKey: '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',
    dangerouslyAllowBrowser:true // Ganti dengan kunci API Anda
  });
  export const getResponse = async (prompt) => {
    try {
      const response = await openai.chat.completions.create({
        model: "gpt-3.5-turbo", // Ganti dengan model yang Anda inginkan
        messages: [{ role: "user", content: prompt }],
      });
      return response.choices[0].message.content; // Mengembalikan konten respons
    } catch (error) {
      console.error("Error fetching response from OpenAI:", error);
      throw error; // Melempar kembali error untuk ditangani di tempat lain
    }
  };
