document.getElementById('file-input').addEventListener('change', function () {
    const previewContainer = document.getElementById('preview-container');
    const previews = document.getElementById('image-previews');
    const files = this.files;
    const fileInput = this;
    
    // Convert FileList to Array for manipulation
    let fileArray = Array.from(files);
    
    if (fileArray.length > 0) {
        previewContainer.classList.remove('hidden');
        document.getElementById('generate-button').disabled = false;
        
        // Clear previous previews
        previews.innerHTML = '';
        
        // Create previews for each file
        fileArray.forEach((file, index) => {
            const reader = new FileReader();
            reader.onload = function (e) {
                const previewWrapper = document.createElement('div');
                previewWrapper.className = 'preview-wrapper';
                previewWrapper.dataset.index = index;
                
                const img = document.createElement('img');
                img.src = e.target.result;
                img.classList.add('preview-image');
                
                const deleteBtn = document.createElement('div');
                deleteBtn.className = 'delete-btn';
                deleteBtn.innerHTML = '×';
                deleteBtn.title = 'Remove image';
                
                // Add delete functionality
                deleteBtn.addEventListener('click', function() {
                    // Remove this preview
                    previewWrapper.remove();
                    
                    // Remove file from our array
                    fileArray.splice(index, 1);
                    
                    // Update other preview indices
                    document.querySelectorAll('.preview-wrapper').forEach((wrapper, idx) => {
                        wrapper.dataset.index = idx;
                    });
                    
                    // Disable generate button if no files left
                    if (fileArray.length === 0) {
                        previewContainer.classList.add('hidden');
                        document.getElementById('generate-button').disabled = true;
                    }
                    
                    // Update the actual file input (requires creating a new DataTransfer)
                    updateFileInput(fileInput, fileArray);
                });
                
                previewWrapper.appendChild(img);
                previewWrapper.appendChild(deleteBtn);
                previews.appendChild(previewWrapper);
            };
            reader.readAsDataURL(file);
        });
    } else {
        previewContainer.classList.add('hidden');
        document.getElementById('generate-button').disabled = true;
    }
});

// Helper function to update file input with new file array
function updateFileInput(fileInput, fileArray) {
    const dataTransfer = new DataTransfer();
    fileArray.forEach(file => {
        dataTransfer.items.add(file);
    });
    fileInput.files = dataTransfer.files;
}

document.getElementById('upload-form').addEventListener('submit', function (e) {
    e.preventDefault();
    const formData = new FormData(this);
    document.getElementById('loader').classList.remove('hidden');
    document.getElementById('results-section').classList.remove('hidden');
    document.getElementById('results-message').innerHTML = '';

    fetch('/upload', {
        method: 'POST',
        body: formData
    }).then(response => response.json())
      .then(data => {
          document.getElementById('loader').classList.add('hidden');
          if (data.error) {
              document.getElementById('results-message').innerHTML = `<p class="error">${data.error}</p>`;
          } else {
              // Display results in the desired order
              const resultsHTML = `
                  <div class="results-container">
                      <h2>Results</h2>
                      <p>${data.message}</p>
                      
                      <div class="results-files">
                          <!-- 1. Captions & Entities -->
                          <div class="result-item">
                              <h3>Captions & Entities</h3>
                              <div class="text-container" id="captions-container">
                                  <p>Loading captions...</p>
                              </div>
                          </div>

                          <!-- 2. Knowledge Graph -->
                          <div class="result-item">
                              <h3>Knowledge Graph</h3>
                              <div class="image-container">
                                  <img src="/${data.output_files.graph}" alt="Knowledge Graph" class="result-image">
                              </div>
                          </div>
                          
                          <!-- 3. Generated Story -->
                          <div class="result-item">
                              <h3>Generated Story</h3>
                              <div class="text-container" id="story-container">
                                  <p>Loading story...</p>
                              </div>
                          </div>

                          <!-- 4. Audio (if available) -->
                          ${data.output_files.audio ? `
                          <div class="result-item">
                              <h3>Listen to the Story</h3>
                              <div class="audio-container">
                                  <audio controls class="audio-player">
                                      <source src="/${data.output_files.audio}" type="audio/wav">
                                      Your browser does not support the audio element.
                                  </audio>
                              </div>
                          </div>
                          ` : ''}
                      </div>
                  </div>
              `;
              
              document.getElementById('results-message').innerHTML = resultsHTML;
              
              // Load the story text
              fetch('/' + data.output_files.story)
                  .then(response => response.text())
                  .then(storyText => {
                      document.getElementById('story-container').innerHTML = `<pre class="story-text">${storyText}</pre>`;
                  });
                  
              // Load the captions text
              fetch('/' + data.output_files.captions)
                  .then(response => response.text())
                  .then(captionsText => {
                      document.getElementById('captions-container').innerHTML = `<pre class="captions-text">${captionsText}</pre>`;
                  });
          }
      }).catch(error => {
          document.getElementById('loader').classList.add('hidden');
          document.getElementById('results-message').innerHTML = `<p class="error">Upload failed: ${error.message}</p>`;
      });
});
