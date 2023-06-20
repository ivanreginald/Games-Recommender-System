function search(inputId, resultsId) {
  var input = document.getElementById(inputId).value;
  var results = document.getElementById(resultsId);
  results.innerHTML = '';

  if (input.trim() === '') {
    return; // Exit the function if the input is empty
  }

  // Send an AJAX request to the Flask route that handles the search
  $.ajax({
    url: '/recommend/game',
    data: { query: input },
    success: function(response) {
      // Update the search results dynamically
      response.forEach(function(game) {
        var li = document.createElement('li');
        li.appendChild(document.createTextNode(game));
        li.addEventListener('click', function() {
          replaceSearchValue(inputId, game);
          hideSearchResults(resultsId);
        });
        results.appendChild(li);
      });

    }
  });
}

function replaceSearchValue(inputId, value) {
  document.getElementById(inputId).value = value;
}

function hideSearchResults(resultsId) {
  var results = document.getElementById(resultsId);
  results.innerHTML = '';
}

