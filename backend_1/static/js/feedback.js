// Feedback Analysis JavaScript
document.addEventListener('DOMContentLoaded', function() {
  // Load particles.js
  particlesJS.load('particles-js', '/static/particles.json', function() {
    console.log('Particles.js config loaded');
  });

  // Fetch feedback data
  fetchFeedbackData();

  // Setup event listeners
  document.getElementById('applyFilters').addEventListener('click', applyFilters);
  document.getElementById('toggleFeedbackBtn').addEventListener('click', toggleFeedbackList);
});

// Fetch feedback data from server
async function fetchFeedbackData() {
  try {
    const response = await fetch('/get_feedback_data');
    const data = await response.json();
    
    if (data) {
      renderDashboard(data);
      populateFeedbackList(data.feedback);
      updateFeedbackCounts(data.feedback);
    }
  } catch (error) {
    console.error('Error fetching feedback data:', error);
  }
}

// Render all dashboard charts
function renderDashboard(data) {
  // Generate dummy data for visualizations
  const dummyData = generateDummyData();
  
  // Render each chart
  renderModelComparisonChart(dummyData.modelComparison);
  renderSatisfactionTrendChart(dummyData.satisfactionTrend);
  renderSuggestedTopics(data.suggested_topics);
  renderResponseTimeChart(dummyData.responseTimes);
  renderUserTypeChart(dummyData.userTypes);
  renderQueryCategoriesChart(dummyData.queryCategories);
}

// Generate dummy data for charts
function generateDummyData() {
  return {
    modelComparison: {
      labels: ['Fine-tuned', 'RAG', 'Llama'],
      datasets: [
        {
          label: 'Avg. Satisfaction',
          data: [4.2, 3.8, 3.5],
          backgroundColor: 'rgba(58, 134, 255, 0.7)',
        },
        {
          label: 'Accuracy Rating',
          data: [4.5, 4.1, 3.7],
          backgroundColor: 'rgba(75, 192, 192, 0.7)',
        }
      ]
    },
    satisfactionTrend: {
      labels: ['Week 1', 'Week 2', 'Week 3', 'Week 4', 'Week 5'],
      datasets: [{
        label: 'User Satisfaction',
        data: [3.5, 3.7, 4.0, 4.2, 4.3],
        borderColor: 'rgba(58, 134, 255, 0.8)',
        backgroundColor: 'rgba(58, 134, 255, 0.2)',
        tension: 0.3,
        fill: true
      }]
    },
    responseTimes: {
      labels: ['Fine-tuned', 'RAG', 'Llama'],
      datasets: [{
        label: 'Avg. Response Time (s)',
        data: [1.2, 2.8, 0.8],
        backgroundColor: 'rgba(153, 102, 255, 0.7)',
      }]
    },
    userTypes: {
      labels: ['Admin', 'Regular User'],
      datasets: [{
        data: [25, 75],
        backgroundColor: ['rgba(58, 134, 255, 0.7)', 'rgba(75, 192, 192, 0.7)'],
      }]
    },
    queryCategories: {
      labels: ['Information', 'Assistance', 'Technical', 'Analysis', 'Other'],
      datasets: [{
        label: 'Query Count',
        data: [45, 30, 22, 18, 10],
        backgroundColor: [
          'rgba(58, 134, 255, 0.7)',
          'rgba(75, 192, 192, 0.7)',
          'rgba(153, 102, 255, 0.7)',
          'rgba(255, 159, 64, 0.7)',
          'rgba(201, 203, 207, 0.7)'
        ],
      }]
    }
  };
}

// Render Model Comparison Chart
function renderModelComparisonChart(data) {
  const ctx = document.getElementById('modelComparisonChart').getContext('2d');
  new Chart(ctx, {
    type: 'bar',
    data: data,
    options: {
      responsive: true,
      scales: {
        y: {
          beginAtZero: true,
          max: 5,
          ticks: {
            color: 'rgba(255, 255, 255, 0.7)'
          },
          grid: {
            color: 'rgba(255, 255, 255, 0.1)'
          }
        },
        x: {
          ticks: {
            color: 'rgba(255, 255, 255, 0.7)'
          },
          grid: {
            color: 'rgba(255, 255, 255, 0.1)'
          }
        }
      },
      plugins: {
        legend: {
          labels: {
            color: 'rgba(255, 255, 255, 0.7)'
          }
        }
      }
    }
  });
}

// Render Satisfaction Trend Chart
function renderSatisfactionTrendChart(data) {
  const ctx = document.getElementById('satisfactionTrendChart').getContext('2d');
  new Chart(ctx, {
    type: 'line',
    data: data,
    options: {
      responsive: true,
      scales: {
        y: {
          beginAtZero: true,
          max: 5,
          ticks: {
            color: 'rgba(255, 255, 255, 0.7)'
          },
          grid: {
            color: 'rgba(255, 255, 255, 0.1)'
          }
        },
        x: {
          ticks: {
            color: 'rgba(255, 255, 255, 0.7)'
          },
          grid: {
            color: 'rgba(255, 255, 255, 0.1)'
          }
        }
      },
      plugins: {
        legend: {
          labels: {
            color: 'rgba(255, 255, 255, 0.7)'
          }
        }
      }
    }
  });
}

// Render Suggested Topics List
function renderSuggestedTopics(topics) {
  const container = document.getElementById('suggestedTopics');
  if (!topics || topics.length === 0) {
    container.innerHTML = '<p class="no-data">No suggested topics available</p>';
    return;
  }

  const topicsList = document.createElement('ul');
  topicsList.className = 'topics-list';
  
  topics.forEach(topic => {
    const item = document.createElement('li');
    item.textContent = topic;
    topicsList.appendChild(item);
  });
  
  container.innerHTML = '';
  container.appendChild(topicsList);
}

// Render Response Time Chart
function renderResponseTimeChart(data) {
  const ctx = document.getElementById('responseTimeChart').getContext('2d');
  new Chart(ctx, {
    type: 'bar',
    data: data,
    options: {
      responsive: true,
      scales: {
        y: {
          beginAtZero: true,
          ticks: {
            color: 'rgba(255, 255, 255, 0.7)'
          },
          grid: {
            color: 'rgba(255, 255, 255, 0.1)'
          }
        },
        x: {
          ticks: {
            color: 'rgba(255, 255, 255, 0.7)'
          },
          grid: {
            color: 'rgba(255, 255, 255, 0.1)'
          }
        }
      },
      plugins: {
        legend: {
          labels: {
            color: 'rgba(255, 255, 255, 0.7)'
          }
        }
      }
    }
  });
}

// Render User Type Chart
function renderUserTypeChart(data) {
  const ctx = document.getElementById('userTypeChart').getContext('2d');
  new Chart(ctx, {
    type: 'doughnut',
    data: data,
    options: {
      responsive: true,
      plugins: {
        legend: {
          position: 'bottom',
          labels: {
            color: 'rgba(255, 255, 255, 0.7)'
          }
        }
      }
    }
  });
}

// Render Query Categories Chart
function renderQueryCategoriesChart(data) {
  const ctx = document.getElementById('queryCategoriesChart').getContext('2d');
  new Chart(ctx, {
    type: 'polarArea',
    data: data,
    options: {
      responsive: true,
      plugins: {
        legend: {
          position: 'bottom',
          labels: {
            color: 'rgba(255, 255, 255, 0.7)'
          }
        }
      },
      scales: {
        r: {
          ticks: {
            color: 'rgba(255, 255, 255, 0.7)'
          },
          grid: {
            color: 'rgba(255, 255, 255, 0.1)'
          }
        }
      }
    }
  });
}

// Update feedback counts display
function updateFeedbackCounts(feedback) {
  if (!feedback) return;
  
  const totalFeedback = feedback.length;
  const positiveFeedback = feedback.filter(f => f.feedback && f.feedback.toLowerCase().includes('good')).length;
  const negativeFeedback = feedback.filter(f => f.feedback && f.feedback.toLowerCase().includes('bad')).length;
  
  const countsElement = document.querySelector('.feedback-counts');
  countsElement.innerHTML = `
    <strong>Total Feedback:</strong> ${totalFeedback} | 
    <strong>Positive:</strong> ${positiveFeedback} | 
    <strong>Negative:</strong> ${negativeFeedback}
  `;
}

// Populate feedback list
function populateFeedbackList(feedback) {
  if (!feedback || feedback.length === 0) return;
  
  const feedbackList = document.getElementById('feedbackList');
  feedbackList.innerHTML = '';
  
  feedback.forEach((item, index) => {
    const feedbackItem = document.createElement('div');
    feedbackItem.className = 'feedback-item';
    feedbackItem.dataset.index = index;
    
    const timestamp = new Date(item.timestamp).toLocaleString();
    const modelType = item.model_type || 'Unknown';
    
    feedbackItem.innerHTML = `
      <div class="feedback-header">
        <strong>${item.email}</strong> (${item.user_type}) - Model: ${modelType} - ${timestamp}
      </div>
      <div class="feedback-details" id="details-${index}">
        <strong>Query:</strong> ${item.query}<br>
        <strong>Response:</strong> <pre>${item.response}</pre><br>
        <strong>Feedback:</strong> ${item.feedback}
      </div>
    `;
    
    feedbackItem.addEventListener('click', function() {
      const detailsElement = document.getElementById(`details-${index}`);
      detailsElement.classList.toggle('show');
    });
    
    feedbackList.appendChild(feedbackItem);
  });
}

// Toggle feedback list visibility
function toggleFeedbackList() {
  const feedbackList = document.getElementById('feedbackList');
  const button = document.getElementById('toggleFeedbackBtn');
  
  if (feedbackList.style.display === 'none') {
    feedbackList.style.display = 'block';
    button.textContent = 'Hide Detailed Feedback';
  } else {
    feedbackList.style.display = 'none';
    button.textContent = 'Show Detailed Feedback';
  }
}

// Apply filters to the dashboard
function applyFilters() {
  const modelType = document.getElementById('modelTypeFilter').value;
  const timeRange = document.getElementById('timeFilter').value;
  
  console.log(`Applying filters: Model Type = ${modelType}, Time Range = ${timeRange}`);
  
  // Ideally, you would re-fetch or filter the data based on these parameters
  // For the demo, we'll just reload the current data
  fetchFeedbackData();
}

// Add CSS for the topics list
const style = document.createElement('style');
style.textContent = `
  .topics-list {
    list-style: none;
    padding: 0;
    margin: 0;
    height: 100%;
    overflow-y: auto;
  }
  
  .topics-list li {
    padding: 10px;
    margin-bottom: 8px;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 8px;
    transition: all 0.3s ease;
  }
  
  .topics-list li:hover {
    background: rgba(255, 255, 255, 0.2);
    transform: translateX(5px);
  }
  
  .no-data {
    color: rgba(255, 255, 255, 0.7);
    text-align: center;
    padding-top: 20%;
  }
`;
document.head.appendChild(style);