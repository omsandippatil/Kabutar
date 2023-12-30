function doGet(e) {
  var emailToCheck = e.parameter.email;
  var sheet = SpreadsheetApp.getActiveSpreadsheet().getActiveSheet();
  var emailColumnValues = sheet.getRange("B:B").getValues().flat(); //As our emails are present in this column
  var isEmailAvailable = emailColumnValues.includes(emailToCheck);
  return ContentService.createTextOutput(isEmailAvailable.toString());
}
