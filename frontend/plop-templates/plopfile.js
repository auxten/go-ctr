/*
plop doc: https://github.com/plopjs/plop
prompt doc: https://github.com/SBoudrias/Inquirer.js/#prompt-types
template doc: https://github.com/wycats/handlebars.js/
*/

const storeGenerator = require('./store/prompt')
const iconGenerator = require('./icons/prompt')

module.exports = function (plop) {
  plop.setGenerator('store', storeGenerator)
  plop.setGenerator('icons', iconGenerator)
}
