// Function to handle summarization
async function handleSummarize() {
  const text = document.getElementById("inputText").value;
  const model = document.getElementById("model").value;

  try {
    document.getElementById('loader-sum').classList.add('show-sum');
    const response = await fetch("/summarize/", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ text, model }),
    });
    
    const result = await response.json();
    document.getElementById('loader-sum').classList.remove('show-sum');
    const statusIcon = document.querySelector("#summarizeStatus");
    
    document.getElementById("summaryOutput").innerText = result.summary || result.error;

    statusIcon.classList.toggle("success", !!result.summary);
    statusIcon.classList.remove("hidden");
    // statusIcon.classList.toggle("error", !result.summary);
  } catch (err) {
    console.error(err);
    const statusIcon = document.querySelector("#summarizeStatus");
    statusIcon.classList.remove("hidden");
    statusIcon.classList.add("error");
    statusIcon.classList.remove("success");
  }
}




// Function to handle translation
async function handleTranslate() {
  const text = document.getElementById("summaryOutput").innerText;
  const target_lang = document.getElementById("targetLang").value;
  document.getElementById('loader-trans').classList.add('show-trans');

  try {
    const response = await fetch("/translate/", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ text, target_lang }),
    });

    const result = await response.json();
    document.getElementById('loader-trans').classList.remove('show-trans');

    document.getElementById("translationOutput").innerText = result.translated_text || result.detail;

    const statusIcon = document.querySelector("#translateStatus");
    statusIcon.classList.remove("hidden");
    statusIcon.classList.toggle("success", !!result.translated_text);
    statusIcon.classList.toggle("error", !result.translated_text);
  } catch (err) {
    console.error(err);
    const statusIcon = document.querySelector("#translateStatus");
    statusIcon.classList.remove("hidden");
    statusIcon.classList.add("error");
    statusIcon.classList.remove("success");
  }
}

// Function to handle PDF upload
async function handleUploadPDF() {
  const fileInput = document.getElementById("pdfFile");
  const formData = new FormData();
  const statusIcon = document.querySelector("#uploadStatus");

  if(fileInput.files.length == 0){
    // console.error(err);
    // statusIcon.classList.toggle("error", true);
    alert("file not uploaded");
    return;
  }
  formData.append("file", fileInput.files[0]);

  document.getElementById('loader-upload').classList.add('show-upload');

  try {
    const response = await fetch("/upload/", {
      method: "POST",
      body: formData,
    });
    

    const result = await response.json();
    // alert(result.message || result.detail);

    alert(result.message || result.detail);
    document.getElementById('loader-upload').classList.remove('show-upload');
    // showStatusIcon("upload", response.ok);
    // const statusIcon = document.querySelector("#uploadStatus");
    // statusIcon.classList.remove("hidden");
    // statusIcon.classList.toggle("success", !!result.translated_text);
    // statusIcon.classList.toggle("error", !result.translated_text);
    console.log(result)
    statusIcon.classList.remove("hidden");
    statusIcon.classList.toggle("success", true);
    // statusIcon.classList.toggle("error", !result.translated_text);
    // statusIcon.classList.toggle("error", !result.message);
  } catch (err) {
    console.error(err);
    statusIcon.classList.toggle("error", true);
    // showStatusIcon("upload", false);
  }
}

// Function to handle question answering
async function handleAskQuestion() {
  const question = document.getElementById("questionInput").value;
  const formData = new FormData();
  formData.append("question", question);
  document.getElementById('loader-ask').classList.add('show-ask');

  try {
    const response = await fetch("/ask/", {
      method: "POST",
      body: formData,
    });

    const result = await response.json();
    document.getElementById('loader-ask').classList.remove('show-ask');

    document.getElementById("answerOutput").innerText = result.answer || result.detail;

    const statusIcon = document.querySelector("#askStatus");
    statusIcon.classList.remove("hidden");
    statusIcon.classList.toggle("success", !!result.answer);
    statusIcon.classList.toggle("error", !result.answer);
  } catch (err) {
    console.error(err);
    const statusIcon = document.querySelector("#askStatus");
    statusIcon.classList.remove("hidden");
    statusIcon.classList.add("error");
    statusIcon.classList.remove("success");
  }
}


