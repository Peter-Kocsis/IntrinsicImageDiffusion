// const SURVEY_ID = 1;

$( document ).ready(function() {

    function alertResults (sender) {
        $.ajax({
            url: "submit",
            type: "get",
            data: sender.data,
        });
    }

    $(function() {
        $.get( "form", function( data ) {
          var survey = new Survey.Model(data);
          survey.onComplete.add(alertResults);
          $("#surveyContainer").Survey({ model: survey });
        });
    });
});
