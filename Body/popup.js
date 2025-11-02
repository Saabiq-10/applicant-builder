document.addEventListener("DOMContentLoaded", () => {
  const results = document.getElementById("results");

  chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
    const tab = tabs[0];

    // Inject content script first
    chrome.scripting.executeScript({
      target: { tabId: tab.id },
      files: ["content.js"]
    }, () => {
      chrome.tabs.sendMessage(tab.id, { action: "getJobDescription" }, async function (response) {
        if (chrome.runtime.lastError) {
          console.error("❌ Messaging failed:", chrome.runtime.lastError.message);
          results.innerText = "❌ Could not extract job description.";
          return;
        }

        if (!response || !response.job) {
          results.innerText = "⚠️ No job info found on this page.";
          return;
        }

        console.log("✅ Got job info:", response);

        const prompt =
          `Job Description:\n${response.job}\n\n` +
          `About the Company:\n${response.company}`;

        results.innerText = "Thinking…";

        try {
            const API_BASE = "http://127.0.0.1:3000"; // local dev backend
            const r = await fetch(`${API_BASE}/generate`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ prompt })
          });

          const data = await r.json();

          if (data.error) {
            results.innerText = "❌ " + data.error;
            return;
          }

          let html = "";
          const section = (title, items) => {
            if (!items || !items.length) return "";
            return `<h3>${title}</h3><ul>` +
              items.map(i => {
                const url = i.url || "#";
                const reason = i.reason || i.why_recommended || "Relevant opportunity.";
                return `<li><a href="${url}" target="_blank">${i.name}</a><br><small>${reason}</small></li>`;
              }).join("") +
              `</ul>`;
          };

          html += section("🛠️ Student Clubs", data.student_teams);
          html += section("🎯 Upcoming Hackathons", data.hackathons);
          html += section("📚 Recommended Courses", data.courses);

          results.innerHTML = html || "No matches found.";
        } catch (e) {
          results.innerText = "❌ Server error.";
          console.error(e);
        }
      });
    });
  });
});
