from flask import Flask,redirect,url_for,render_template,request
import pickle
app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def welcome():
    return render_template('homepage.html')

@app.route('/submit',methods=['POST','GET'])
def submit():
    popularity=0
    if request.method=='POST':
        duration_ms=float(request.form['duration_ms'])
        danceability=float(request.form['danceability'])
        energy=float(request.form['energy'])
        key=float(request.form['key'])
        loudness=float(request.form['loudness'])
        speechiness=float(request.form['speechiness'])
        acousticness=float(request.form['acousticness'])
        instrumentalness=float(request.form['instrumentalness'])
        liveness=float(request.form['liveness'])
        valence=float(request.form['valence'])
        tempo=float(request.form['tempo'])
        release_year=float(request.form['release_year'])
        release_month=float(request.form['release_month'])
        release_day=float(request.form['release_day'])
        
    popularity=model.predict([[duration_ms,danceability,energy,key,loudness,speechiness,acousticness,instrumentalness,liveness,valence,tempo,release_year,release_month,release_day
]])

    return redirect(url_for('result',POPULARITY=popularity)) 
        
   
@app.route('/result,<int:POPULARITY>')
def result(POPULARITY):
    out=POPULARITY



    return render_template('result_spotify.html',output=out)          

if __name__=='__main__':
    app.run(debug=True)    
