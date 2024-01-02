function doGet(e) {
  var email = e.parameter.email;
  var password = e.parameter.password;
  var displayName = e.parameter.displayname;
  var bio = e.parameter.bio;
  var handle = e.parameter.handle;
  var tstamp = new Date();
  var sheet = SpreadsheetApp.getActiveSpreadsheet().getActiveSheet();
  var lastRow = sheet.getLastRow();
  sheet.appendRow([ lastRow,email, password, displayName, bio, tstamp ,handle]);
  return ContentService.createTextOutput("successful");
}
