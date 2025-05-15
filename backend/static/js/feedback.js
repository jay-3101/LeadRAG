let allFeedbacks = [];
let showingAll = false;


const stopwords = ["the", "is", "in", "and", "to", "a", "of", "for", "on", "it", "this", "that", "can", "be", "how"];

async function loadFeedback() {
  const type = document.getElementById('typeFilter').value;
  const model = document.getElementById('modelFilter').value;
  
  const res = await fetch('/get_feedback_data');
  const data = await res.json();

  const feedbackArray = data.feedback || [];  // ✅ fixed: access actual array
  const suggestedTopics = data.suggested_topics || [];  // ✅ topic data

  // Filtering logic based on user type and model type
  const filtered = feedbackArray.filter(entry => {
    const typeMatch = type === 'all' || entry.user_type === type;
    const normalizedModel = (entry.model_type || '').toLowerCase();
    const modelMatch = model === 'all' || normalizedModel === model;  
    return typeMatch && modelMatch;
  });

  // Update total counts for feedback
  const total = filtered.length;
  const userCount = filtered.filter(f => f.user_type === 'user').length;
  const adminCount = filtered.filter(f => f.user_type === 'admin').length;

  // Update the display with counts
  document.getElementById("totalCount").innerText = `Total Feedbacks: ${total}`;
  document.getElementById("userCount").innerText = `From Users: ${userCount}`;
  document.getElementById("adminCount").innerText = `From Admins: ${adminCount}`;

  // Call renderFeedbackList instead of displayFeedback
  renderFeedbackList(filtered);

  // Word Frequency Calculation
  const queries = filtered.map(f => f.query);
  const wordFreq = getWordFrequencies(queries);
  

  renderModelComparisonChart(data.model_performance);
  renderSatisfactionTrendChart(data.satisfaction_trend);
  renderResponseTimeChart(data.response_time);
  renderUserTypeChart(data.user_type_distribution);
  renderQueryCategoryChart(data.query_categories);
  renderDocSuggestChart(data.suggested_topics);

}


function getWordFrequencies(queries) {
  const counts = {};
  queries.forEach(q => {
    q.toLowerCase().split(/\W+/).forEach(word => {
      if (word && !stopwords.includes(word)) {
        counts[word] = (counts[word] || 0) + 1;
      }
    });
  });
  return counts;
}

let chart;

function renderDocSuggestChart(topics) {
  const ctx = document.getElementById('docSuggestChart').getContext('2d');

  if (!topics.length) return;

  new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: topics,
      datasets: [{
        data: Array(topics.length).fill(1),
        backgroundColor: ['#ff4d4d', '#ff6666', '#ff9999', '#cc0000', '#ff3333']
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: '#fff' } }
      }
    }
  });
}

function renderModelComparisonChart(data) {
  if (!data || Object.keys(data).length === 0) return;

  const ctx = document.getElementById('modelComparisonChart').getContext('2d');
  const models = Object.keys(data);
  const scores = Object.values(data);

  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: models,
      datasets: [{
        label: 'Average Score',
        data: scores,
        backgroundColor: 'rgba(54, 162, 235, 0.7)',
      }]
    },
    options: {
      responsive: true,
      plugins: {
        legend: { display: false }
      },
      scales: {
        y: { beginAtZero: true, ticks: { color: '#fff' } },
        x: { ticks: { color: '#fff' } }
      }
    }
  });
}

function renderSatisfactionTrendChart(data) {
  const ctx = document.getElementById('satisfactionTrendChart').getContext('2d');

  // Limit to first 4 weeks
  const maxWeeks = 4;
  const labels = data.dates.slice(0, maxWeeks).map((_, idx) => `Week ${idx + 1}`);
  const scores = data.scores.slice(0, maxWeeks);

  new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [{
        label: 'Satisfaction',
        data: scores,
        fill: false,
        borderColor: '#4caf50',  // soft green line
        backgroundColor: '#4caf50',
        tension: 0.2,
        pointRadius: 5,
        pointHoverRadius: 7,
        borderWidth: 2
      }]
    },
    options: {
      responsive: true,
      plugins: {
        legend: {
          labels: { color: '#fff' }
        }
      },
      scales: {
        y: {
          ticks: { color: '#fff' },
          beginAtZero: true,
          max: 5 // Assuming satisfaction scale is 0-5
        },
        x: {
          ticks: { color: '#fff' }
        }
      }
    }
  });
}



function renderResponseTimeChart(data) {
  if (!data) return;

  const ctx = document.getElementById('responseTimeChart').getContext('2d');
  const models = Object.keys(data);
  const times = Object.values(data);

  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: models.map(m => m.toUpperCase()),
      datasets: [{
        label: 'Avg Response Time (ms)',
        data: times,
        backgroundColor: ['#03a9f4', '#8bc34a', '#ff9800']
      }]
    },
    options: {
      responsive: true,
      plugins: {
        legend: { display: false }
      },
      scales: {
        x: { ticks: { color: '#fff' } },
        y: {
          beginAtZero: true,
          ticks: { color: '#fff' }
        }
      }
    }
  });
}

function renderUserTypeChart(counts) {
  const ctx = document.getElementById('userTypeChart').getContext('2d');

  new Chart(ctx, {
    type: 'pie',
    data: {
      labels: Object.keys(counts),
      datasets: [{
        data: Object.values(counts),
        backgroundColor: [
  '#4caf50', // green - User
  '#f44336', // red - Admin
  '#2196f3'  // blue - (optional third category)
]
      }]
    },
    options: {
      plugins: {
        legend: { labels: { color: '#fff' } }
      }
    }
  });
}

function renderQueryCategoryChart(categories) {
  const ctx = document.getElementById('queryCategoriesChart').getContext('2d');

  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: Object.keys(categories),
      datasets: [{
        label: 'Query Count',
        data: Object.values(categories),
        backgroundColor: 'teal'
      }]
    },
    options: {
      responsive: true,
      plugins: {
        legend: { display: false }
      },
      scales: {
        x: { ticks: { color: '#fff' } },
        y: { ticks: { color: '#fff' } }
      }
    }
  });
}


function renderFeedbackList(data) {
  allFeedbacks = data;
  const listDiv = document.getElementById("feedbackList");
  const maxInitial = 5;

  listDiv.innerHTML = "";
  const visibleFeedbacks = showingAll ? data : data.slice(0, maxInitial);

  visibleFeedbacks.forEach((item, idx) => {
    const card = document.createElement("div");
    card.className = "feedback-item";
    card.innerHTML = `
        <div class="feedback-header">
            ${idx + 1}. <strong>Query:</strong> ${item.query}<br>
            <strong>Feedback:</strong> ${item.feedback}
        </div>
      
      <div class="feedback-details">
        <p><strong>Feedback:</strong> ${item.feedback}</p>
        <p><strong>Email:</strong> ${item.email}</p>
        <p><strong>User Type:</strong> ${item.user_type}</p>
        <p><strong>Model Type:</strong> ${item.model_type}</p>
        <p><strong>Response:</strong></p>
        <pre>${item.response}</pre>
        <p><strong>Time:</strong> ${item.timestamp}</p>
      </div>
    `;
    card.onclick = () => {
      const detail = card.querySelector(".feedback-details");
      detail.classList.toggle("show");
    };
    listDiv.appendChild(card);
  });

  document.getElementById("toggleFeedbackBtn").style.display = data.length > maxInitial ? "block" : "none";
  document.getElementById("toggleFeedbackBtn").innerText = showingAll ? "Show Less" : "Show More";
}

function toggleFeedbackVisibility() {
  showingAll = !showingAll;
  renderFeedbackList(allFeedbacks);
}


// Initial load
loadFeedback();
