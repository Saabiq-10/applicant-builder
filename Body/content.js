chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === "getJobDescription") {
    console.log("📨 Received request in content.js");

    const allText = document.body.innerText;
    const jobMatch = allText.match(/About the job[\s\S]*/i);
    const companyMatch = allText.match(/About the company[\s\S]*?(?=\n\n|\r\n\r\n|$)/i);

    const job = jobMatch ? jobMatch[0].replace(/About the job/i, "").trim() : "";
    const company = companyMatch ? companyMatch[0].replace(/About the company/i, "").trim() : "";

    console.log("📄 Extracted job:", job.slice(0, 100));
    console.log("🏢 Extracted company:", company.slice(0, 100));

    sendResponse({ job, company });
  }
  return true;
});
