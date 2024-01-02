function emailExists(email, sheet) {
  var emailColumn = getColumnIndex(sheet, 'Email');
  var emails = sheet.getRange(2, emailColumn, sheet.getLastRow() - 1, 1).getValues();

  for (var i = 0; i < emails.length; i++) {
    if (emails[i][0] == email) {
      return true;
    }
  }

  return false;
}

function addDataToSheet(email, password, displayname, bio, biolink) {
  var sheet = SpreadsheetApp.getActiveSpreadsheet().getActiveSheet();

  // Checq if the email already exists
  if (emailExists(email, sheet)) {
    return 'error: Email already exists';
  }

  var lastRow = sheet.getLastRow();
  var tstamp = new Date();

  sheet.getRange(lastRow + 1, 1).setValue(lastRow); //using it for user id
  sheet.getRange(lastRow + 1, 2).setValue(email);
  sheet.getRange(lastRow + 1, 3).setValue(password);
  sheet.getRange(lastRow + 1, 4).setValue(displayname);
  sheet.getRange(lastRow + 1, 5).setValue(bio);
  sheet.getRange(lastRow + 1, 6).setValue(biolink);
  sheet.getRange(lastRow + 1, 7).setValue(tstamp);

  return 'success';
}

// Google Apps Script web app endpoint
function doGet(e) {
  var email = e.parameter.email;
  var password = e.parameter.password;
  var displayname = e.parameter.displayname;
  var handle = e.parameter.handle;
  var bio = e.parameter.bio;
  var biolink = e.parameter.biolink;

  var result = addDataToSheet(email, password, displayname, handle, bio, biolink);
  return ContentService.createTextOutput(JSON.stringify(result)).setMimeType(ContentService.MimeType.JSON);
}

function getColumnIndex(sheet, columnName) {
  var headers = sheet.getRange(1, 1, 1, sheet.getLastColumn()).getValues()[0];
  for (var i = 0; i < headers.length; i++) {
    if (headers[i] == columnName) {
      return i + 1; // Columns in Google Sheets are 1-indexed
    }
  }
  return -1; // Column not found
}
