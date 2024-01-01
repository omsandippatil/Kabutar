function getDataFromSpreadsheet() {

  var spreadsheetId = 'Your_Sheet_ID';
  var sheetName = 'Sheet_Name';
  var spreadsheet = SpreadsheetApp.openById(spreadsheetId);
  var sheet = spreadsheet.getSheetByName(sheetName);
range
  var dataRange = sheet.getDataRange();
  var values = dataRange.getValues();

  var headers = values[0];
  var emailIndex = headers.indexOf('Email');
  var displayNameIndex = headers.indexOf('display_name');
  var bioIndex = headers.indexOf('bio');
  var handleIndex = headers.indexOf('handle');
  var photoIndex = headers.indexOf('photo');
  var verifiedIndex = headers.indexOf('verified');


  for (var i = 1; i < values.length; i++) {
    var row = values[i];
    var email = row[emailIndex];
    var displayName = row[displayNameIndex];
    var bio = row[bioIndex];
    var handle = row[handleIndex];
    var photo = row[photoIndex];
    var verified = row[verifiedIndex];

    Logger.log('Email: ' + email +
               ', Display Name: ' + displayName +
               ', Bio: ' + bio +
               ', Handle: ' + handle +
               ', Photo: ' + photo +
               ', Verified: ' + verified);
  }
}
