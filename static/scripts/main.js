function getBotResponse() {
    let user_image_str = "<img src=\"/static/images/user.png\" width=\"20px\" height=\"20px\" alt=\"CANDICE\" style=\"width:50px;height:50px;\"></img>";
    
    let rawText = $("#textInput").val();
    let userHtml =  '<p class="userText"><span> xxxxx' + user_image_str + ': ' + rawText + "</span></p>";
    $("#textInput").val("");
    $("#chatbox").append(userHtml);
    document.getElementById("userInput").scrollIntoView({ block: "start", behavior: "smooth" });

    $.get("/get", { msg: rawText }).done(function (data) {
      var res = data.split("@@@");
      if(res.length == 1){
        var botHtml = '<p class="botText"><span>' + "The last query has been deleted." + "</span></p>";
        $("#chatbox").append(botHtml);
      }
      else{
        var rewrite = "Conversational Query Rewrite: " + res[0];
        var rewrite_botHtml = '<p class="botText"><span>' + rewrite + "</span></p>";
        $("#chatbox").append(rewrite_botHtml);

        var doc_content = "Passage Content: " + res[1];
        var doc_botHtml = '<p class="botText"><span>' + doc_content + "</span></p>";
        $("#chatbox").append(doc_botHtml);
        
        if(res.length == 4){
          var clarification_q = "Query Clarification: " + res[2];
          var clarify_botHtml = '<p class="botText"><span>' + clarification_q + "</span></p>";
          $("#chatbox").append(clarify_botHtml);
          
          var clari_candidates = "Possible Candidates: " + res[3];
          var candidate_botHtml = '<p class="botText"><span>' + clari_candidates + "</span></p>";
          $("#chatbox").append(candidate_botHtml);
        }
      }
    
      document
        .getElementById("userInput")
        .scrollIntoView({ block: "start", behavior: "smooth" });
    });
    
  }
  $("#textInput").keypress(function (e) {
    if (e.which == 13) {
      getBotResponse();
    }
  });